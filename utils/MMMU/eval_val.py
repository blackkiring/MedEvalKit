import torch
import os
import random
import json

import numpy as np
from tqdm import tqdm

from datasets import load_dataset, concatenate_datasets

from argparse import ArgumentParser
from mathruler.grader import extract_boxed_content
from .data_utils import load_yaml, construct_prompt, save_json, process_single_sample, CAT_SHORT2LONG,DOMAIN_CAT2SUB_CAT
from .eval_utils import evaluate,parse_multi_choice_response, parse_open_response
from ..utils import extract

def _load_existing_samples(output_file):
    """Load existing processed samples if file exists."""
    if not os.path.exists(output_file):
        return None
    
    try:
        with open(output_file, "r") as f:
            existing_samples = json.load(f)
        if isinstance(existing_samples, list) and len(existing_samples) > 0:
            # Check if samples have responses
            if all("response" in s for s in existing_samples):
                return existing_samples
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not load existing samples from {output_file}: {e}")
    
    return None

def run_model(samples, model, existing_samples=None):
    """Run model on samples, skipping already processed ones if existing_samples provided."""
    # If we have existing samples, filter out already-processed ones
    if existing_samples:
        existing_by_id = {s["id"]: s for s in existing_samples if "response" in s}
        remaining_samples = [s for s in samples if s["id"] not in existing_by_id]
        
        if len(remaining_samples) < len(samples):
            print(f"Resume: Skipping {len(samples) - len(remaining_samples)} already-processed samples")
            print(f"Resume: Processing {len(remaining_samples)} remaining samples")
        
        samples_to_process = remaining_samples
    else:
        samples_to_process = samples
        existing_by_id = {}
    
    new_out_samples = []
    if not samples_to_process:
        print("All samples already processed. Skipping inference.")
        return existing_samples if existing_samples else []
    
    with torch.no_grad():
        messages_list = []
        current_messages = []
        current_samples = []
        for sample in tqdm(samples_to_process):
            messages = {"prompt":sample["final_input_prompt"],"images":sample["images"]}
            current_messages.append(messages)
            current_samples.append(sample)
            if len(current_messages) >= 2000:
                messages_list.append([current_messages,current_samples])
                current_messages = []
                current_samples = []
        if current_messages:
            messages_list.append([current_messages,current_samples])

        for current_messages,current_samples in tqdm(messages_list):
            outputs = model.generate_outputs(current_messages)
            for sample,response in zip(current_samples,outputs):
                out_sample = {
                    "id":sample['id'],
                    "question_type":sample['question_type'],
                    "answer":sample["answer"]
                }
                if "<answer>" in response:
                    response = extract(response,"answer")
                if extract_boxed_content(response) != "None":
                    response = extract_boxed_content(response)

                if sample['question_type'] == 'multiple-choice':
                    out_sample["all_choices"] = sample["all_choices"]
                    out_sample["index2ans"] = sample["index2ans"]
                    out_sample["response"] = response
                    out_sample["parsed_pred"] = parse_multi_choice_response(response, sample['all_choices'], sample['index2ans'])
                else:  # open question
                    out_sample["response"] = response
                    out_sample["parsed_pred"] = response
                new_out_samples.append(out_sample)
    
    # Combine in the original sample order
    if existing_by_id:
        # Build dict of new results by ID
        new_by_id = {s["id"]: s for s in new_out_samples}
        # Reconstruct in original order
        out_samples = []
        for sample in samples:
            sample_id = sample["id"]
            if sample_id in existing_by_id:
                out_samples.append(existing_by_id[sample_id])
            elif sample_id in new_by_id:
                out_samples.append(new_by_id[sample_id])
        return out_samples
    
    return new_out_samples


def eval_MMMU_val(model,dataset_path,output_path,subset):
    total_results = {"total":{"total":0,"right":0}}
    for subject in tqdm(DOMAIN_CAT2SUB_CAT[subset]):
        sub_samples = []
        sub_dataset = load_dataset(dataset_path, subject, split="validation")

        eval_sub_path = os.path.join(output_path,subject)
        if not os.path.exists(eval_sub_path):
            os.mkdir(eval_sub_path)
        for sample in sub_dataset:
            sample = process_single_sample(sample)
            sample = construct_prompt(sample)
            sub_samples.append(sample)

        # Check for existing results to enable resume
        output_sample_path = os.path.join(eval_sub_path,"output_sample.json")
        existing_samples = _load_existing_samples(output_sample_path)
        if existing_samples:
            print(f"Found {len(existing_samples)} existing samples for {subject}")
        
        eval_samples = run_model(sub_samples, model, existing_samples)
        save_json(output_sample_path, eval_samples)
        judge_dict, metric_dict = evaluate(eval_samples)
        metric_dict.update({"num_example": len(eval_samples)})
        for eval_sample in eval_samples:
            eval_sample.update({"judge": judge_dict[eval_sample['id']]})

        save_json(os.path.join(eval_sub_path, 'parsed_output.json'), eval_samples)
        save_json(os.path.join(eval_sub_path, 'result.json'), metric_dict)
        total_results[subject] = metric_dict
        total_results["total"]["total"] += metric_dict["total"]
        total_results["total"]["right"] += metric_dict["right"]
    total_results["total"]["acc"] = total_results["total"]["right"] / total_results["total"]["total"]
    save_json(os.path.join(output_path, 'result.json'), total_results)
    return total_results
