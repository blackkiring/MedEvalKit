#!/usr/bin/env python
"""
Medical Report Metrics Calculator

Computes evaluation metrics for medical report generation tasks.
Basic metrics (BLEU, ROUGE, METEOR) are always computed.
Advanced metrics (CIDEr, BERTScore, RaTEScore, GREEN) are optional.

Usage:
    uv run python utils/Metrics_Compute/cal_report_metrics.py --model_path eval_results/Qwen/Qwen2.5-VL-7B-Instruct

    # With advanced metrics (requires additional packages)
    uv run python utils/Metrics_Compute/cal_report_metrics.py --model_path eval_results/Qwen/Qwen2.5-VL-7B-Instruct --advanced
"""

import os
import json
import argparse
from tqdm import tqdm

# Basic dependencies (always required)
import nltk
from nltk.translate.meteor_score import single_meteor_score
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

# Download NLTK data if not present
try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('omw-1.4')
except LookupError:
    nltk.download('omw-1.4', quiet=True)


def prep_reports(reports):
    """Preprocesses reports for BLEU/METEOR computation"""
    return [list(filter(
        lambda val: val != "", str(elem).lower().replace(".", " .").split(" ")))
        for elem in reports]


def compute_basic_metrics(datas):
    """Compute basic metrics: BLEU, ROUGE, METEOR"""
    rouge_scorer = Rouge()

    total_bleu1 = 0
    total_bleu2 = 0
    total_bleu3 = 0
    total_bleu4 = 0
    total_rouge1 = 0
    total_rouge2 = 0
    total_rougel = 0
    total_meteor = 0

    reports = []
    preds = []
    valid_count = 0

    for sample in tqdm(datas, desc="Computing basic metrics"):
        response = sample.get("response", "")
        if not response or response.strip() == "":
            continue

        findings = sample.get("findings", "")
        impression = sample.get("impression", "")
        golden = f"Findings: {findings} Impression: {impression}."

        tokenized_response = prep_reports([response.lower()])[0]
        tokenized_golden = prep_reports([golden.lower()])[0]

        if not tokenized_response or not tokenized_golden:
            continue

        valid_count += 1
        reports.append(golden)
        preds.append(response)

        # BLEU scores
        try:
            bleu1 = sentence_bleu([tokenized_golden], tokenized_response, weights=[1])
            bleu2 = sentence_bleu([tokenized_golden], tokenized_response, weights=[0.5, 0.5])
            bleu3 = sentence_bleu([tokenized_golden], tokenized_response, weights=[1/3, 1/3, 1/3])
            bleu4 = sentence_bleu([tokenized_golden], tokenized_response, weights=[0.25, 0.25, 0.25, 0.25])
            total_bleu1 += bleu1
            total_bleu2 += bleu2
            total_bleu3 += bleu3
            total_bleu4 += bleu4
        except Exception as e:
            print(f"BLEU error: {e}")

        # ROUGE scores
        try:
            rouge_scores = rouge_scorer.get_scores(response.lower(), golden.lower())
            total_rouge1 += rouge_scores[0]["rouge-1"]["f"]
            total_rouge2 += rouge_scores[0]["rouge-2"]["f"]
            total_rougel += rouge_scores[0]["rouge-l"]["f"]
        except Exception as e:
            print(f"ROUGE error: {e}")

        # METEOR score
        try:
            meteor = single_meteor_score(reference=tokenized_golden, hypothesis=tokenized_response)
            total_meteor += meteor
        except Exception as e:
            print(f"METEOR error: {e}")

    if valid_count == 0:
        return {}, [], []

    metrics = {
        "bleu1": float(total_bleu1 / valid_count),
        "bleu2": float(total_bleu2 / valid_count),
        "bleu3": float(total_bleu3 / valid_count),
        "bleu4": float(total_bleu4 / valid_count),
        "rouge1": float(total_rouge1 / valid_count),
        "rouge2": float(total_rouge2 / valid_count),
        "rougeL": float(total_rougel / valid_count),
        "meteor": float(total_meteor / valid_count),
        "num_samples": valid_count,
    }

    return metrics, reports, preds


def compute_advanced_metrics(reports, preds):
    """Compute advanced metrics: CIDEr, BERTScore, RaTEScore, GREEN"""
    advanced_metrics = {}

    # CIDEr
    try:
        from cidereval import cider
        print("Computing CIDEr score...")
        reports2 = [[r.lower()] for r in reports]
        cider_score = cider(predictions=[p.lower() for p in preds], references=reports2)
        advanced_metrics["cider"] = float(cider_score["avg_score"])
        print(f"  CIDEr: {advanced_metrics['cider']:.4f}")
    except ImportError:
        print("  CIDEr: skipped (cidereval not installed)")
    except Exception as e:
        print(f"  CIDEr error: {e}")

    # BERTScore
    try:
        from bert_score import BERTScorer
        print("Computing BERTScore...")
        scorer = BERTScorer(lang="en", rescale_with_baseline=True)
        P, R, F1 = scorer.score(preds, reports)
        advanced_metrics["bertscore_p"] = float(P.mean())
        advanced_metrics["bertscore_r"] = float(R.mean())
        advanced_metrics["bertscore_f1"] = float(F1.mean())
        print(f"  BERTScore F1: {advanced_metrics['bertscore_f1']:.4f}")
    except ImportError:
        print("  BERTScore: skipped (bert_score not installed)")
    except Exception as e:
        print(f"  BERTScore error: {e}")

    # RaTEScore
    try:
        from RaTEScore import RaTEScore
        print("Computing RaTEScore...")
        ratescore = RaTEScore(bert_model="Angelakeke/RaTE-NER-Deberta", eval_model='FremyCompany/BioLORD-2023-C')
        rate_scores = ratescore.compute_score(preds, reports)
        advanced_metrics["ratescore"] = float(sum(rate_scores) / len(rate_scores))
        print(f"  RaTEScore: {advanced_metrics['ratescore']:.4f}")
    except ImportError:
        print("  RaTEScore: skipped (RaTEScore not installed)")
    except Exception as e:
        print(f"  RaTEScore error: {e}")

    # GREEN Score
    try:
        from green_score import GREEN
        print("Computing GREEN score...")
        green_scorer = GREEN("GREEN-radllama2-7b", output_dir=".")
        green_mean, green_std, _, _, _ = green_scorer(reports, preds)
        advanced_metrics["green_mean"] = float(green_mean)
        advanced_metrics["green_std"] = float(green_std)
        advanced_metrics["green_score"] = f"{green_mean:.4f} ± {green_std:.4f}"
        print(f"  GREEN: {advanced_metrics['green_score']}")
    except ImportError:
        print("  GREEN: skipped (green_score not installed)")
    except Exception as e:
        print(f"  GREEN error: {e}")

    return advanced_metrics


def process_dataset(model_path, dataset, compute_advanced=False):
    """Process a single dataset"""
    results_path = os.path.join(model_path, dataset, "results.json")
    output_path = os.path.join(model_path, dataset, "report_metrics.json")

    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return None

    print(f"\n{'='*60}")
    print(f"Processing: {dataset}")
    print(f"{'='*60}")

    with open(results_path, "r", encoding='utf-8') as f:
        datas = json.load(f)

    print(f"Loaded {len(datas)} samples")

    # Compute basic metrics
    metrics, reports, preds = compute_basic_metrics(datas)

    if not metrics:
        print("No valid samples found!")
        return None

    # Compute advanced metrics if requested
    if compute_advanced and reports:
        advanced = compute_advanced_metrics(reports, preds)
        metrics.update(advanced)

    # Print results
    print(f"\n--- Results for {dataset} ---")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Save metrics
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    print(f"\nMetrics saved to: {output_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Compute medical report generation metrics")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model results directory (e.g., eval_results/Qwen/Qwen2.5-VL-7B-Instruct)")
    parser.add_argument("--datasets", type=str, nargs="+",
                       default=["IU_XRAY", "MIMIC_CXR", "CheXpert_Plus"],
                       help="Datasets to process")
    parser.add_argument("--advanced", action="store_true",
                       help="Compute advanced metrics (CIDEr, BERTScore, RaTEScore, GREEN)")

    args = parser.parse_args()

    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ['PYTHONIOENCODING'] = 'utf-8'

    all_metrics = {}
    for dataset in args.datasets:
        metrics = process_dataset(args.model_path, dataset, args.advanced)
        if metrics:
            all_metrics[dataset] = metrics

    # Save summary
    if all_metrics:
        summary_path = os.path.join(args.model_path, "report_metrics_summary.json")
        with open(summary_path, "w", encoding='utf-8') as f:
            json.dump(all_metrics, f, indent=4, ensure_ascii=False)
        print(f"\n{'='*60}")
        print(f"Summary saved to: {summary_path}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
