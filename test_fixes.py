#!/usr/bin/env python3
"""Quick test to verify fixes"""

import os
os.environ['REASONING'] = 'False'

def construct_prompt(sample):
    """Copy of the updated construct_prompt function"""
    question = sample['question']
    options = eval(sample['options'])
    
    # Get number of images if available
    num_images = len(sample.get('images', []))
    
    # Construct image index information
    if num_images == 0:
        image_index_info = ""
    elif num_images == 1:
        image_index_info = "The index of the given image is 1.\n"
    else:
        indices = ", ".join(str(i) for i in range(1, num_images + 1))
        image_index_info = f"The indices of the given images are {indices}.\n"
    
    example = ""
    if sample['question_type'] == 'multiple-choice':
        start_chr = 'A'
        prediction_range = []
        index2ans = {}
        options_list = []
        for option in options:
            prediction_range.append(start_chr)
            options_list.append(f"({start_chr}) {option}")
            index2ans[start_chr] = option
            start_chr = chr(ord(start_chr) + 1)
        example = " ".join(options_list)
        
        # Construct prompt with new format
        empty_prompt = f"### Question:\n{question}\nOptions: {example}\n{image_index_info}"

        if os.environ["REASONING"] == "True":
            empty_prompt += 'Answer with the option\'s letter from the given choices and put the letter in one "\\boxed{}".'
        else:
            empty_prompt += "Answer with the option's letter from the given choices directly." 

        res_dict = {}
        res_dict['index2ans'] = index2ans
        res_dict['correct_choice'] = sample['answer']
        res_dict['all_choices'] = prediction_range
        res_dict['empty_prompt'] = empty_prompt
        res_dict['final_input_prompt'] = empty_prompt
        res_dict['gt_content'] = options[ord(sample['answer'].upper()) - ord('A')]
    else:
        # Construct prompt with new format for open questions
        empty_prompt = f"### Question:\n{question}\n{image_index_info}"

        if os.environ["REASONING"] == "True":
            empty_prompt += 'Answer the question using a single word or phrase and put the answer in one "\\boxed{}".'
        else:
            empty_prompt += "Answer the question using a single word or phrase." 

        res_dict = {}
        res_dict['empty_prompt'] = empty_prompt
        res_dict['final_input_prompt'] = empty_prompt
        res_dict['gt_content'] = sample['answer']

    res_dict.update(sample)
    return res_dict

# Test with 3 images
sample = {
    'question': 'Compare these images',
    'options': "['Option A', 'Option B', 'Option C']",
    'answer': 'A',
    'question_type': 'multiple-choice',
    'images': ['/img1.jpg', '/img2.jpg', '/img3.jpg']
}

result = construct_prompt(sample)
print("Test - Multiple Images:")
print("="*80)
print(result['final_input_prompt'])
print("="*80)

assert "indices of the given images are 1, 2, 3" in result['final_input_prompt'], "Grammar fixed"
assert not result['final_input_prompt'].endswith(" \n"), "No trailing space before newline"
print("✓ Grammar and spacing issues fixed!")
