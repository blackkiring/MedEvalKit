# Breakpoint Resume Support

## Overview

The MedEvalKit evaluation pipeline now supports resuming from previously saved results. If an evaluation is interrupted (e.g., due to system crash, manual interruption, or timeout), you can simply re-run the same command, and the evaluation will automatically resume from where it left off.

## How It Works

### Detection of Existing Results

When you run an evaluation, the system automatically:
1. Checks for existing `results.json` or `results_{chunk_idx}.json` files in the output directory
2. Loads previously completed samples
3. Identifies which samples have already been processed (by matching sample IDs or content)
4. Skips completed samples and only processes remaining ones
5. Merges existing and new results
6. Recalculates metrics on the complete dataset
7. Updates `total_results.json` incrementally

### Resume Logging

When resuming, you'll see messages like:
```
Found existing results with 150 samples at /path/to/results.json
Resume: Skipping 150 already-processed samples
Resume: Processing 50 remaining samples
```

This helps you understand what work is being skipped vs. what still needs to be done.

## Usage

### Single-Chunk Evaluation

Simply re-run the same evaluation command:

```bash
python eval.py \
    --eval_datasets MMMU-Medical-val \
    --output_path eval_results/my_model \
    --model_name my_model \
    --model_path path/to/model
```

If the evaluation was interrupted:
- The system will load `eval_results/my_model/MMMU-Medical-val/results.json`
- Skip samples that already have responses
- Process only the remaining samples
- Update the results with complete data

### Multi-Chunk Evaluation

For distributed evaluation with chunks:

```bash
# Terminal 1 - Run chunk 0
python eval.py \
    --eval_datasets PATH_VQA \
    --output_path eval_results/my_model \
    --model_name my_model \
    --model_path path/to/model \
    --num_chunks 4 \
    --chunk_idx 0
```

If chunk 0 is interrupted:
- Re-run the same command
- The system will load `eval_results/my_model/PATH_VQA/results_0.json`
- Skip completed samples
- Process remaining samples

The system will automatically merge results and calculate metrics when all chunks complete.

### Multiple Datasets

When evaluating multiple datasets:

```bash
python eval.py \
    --eval_datasets MMMU-Medical-val,PATH_VQA,VQA_RAD \
    --output_path eval_results/my_model \
    --model_name my_model \
    --model_path path/to/model
```

If interrupted after completing MMMU-Medical-val:
- Re-run the same command
- MMMU-Medical-val will be skipped (all samples already complete)
- PATH_VQA and VQA_RAD will be processed
- `total_results.json` will be updated incrementally

## File Structure

### Single-Chunk Mode

```
eval_results/my_model/
├── DATASET_NAME/
│   ├── results.json      # All sample results (updated incrementally)
│   └── metrics.json      # Calculated metrics
└── total_results.json    # Aggregated results across datasets
```

### Multi-Chunk Mode

```
eval_results/my_model/
├── DATASET_NAME/
│   ├── results_0.json    # Chunk 0 results (updated incrementally)
│   ├── results_1.json    # Chunk 1 results (updated incrementally)
│   ├── results_N.json    # Chunk N results (updated incrementally)
│   ├── results.json      # Merged results (created when all chunks complete)
│   └── metrics.json      # Calculated metrics (created when all chunks complete)
└── total_results.json    # Aggregated results across datasets
```

## Edge Cases

### Corrupted Result Files

If a result file is corrupted (invalid JSON):
- The system will print a warning
- Treat it as if no results exist
- Start processing from the beginning

### Partially Complete Chunks

If you have:
- `results_0.json` - complete
- `results_1.json` - partial (only 50 of 100 samples)
- `results_2.json` - missing

Re-running will:
- Skip chunk 0 entirely (all samples complete)
- Resume chunk 1 (process remaining 50 samples)
- Process chunk 2 from scratch

### Missing Sample IDs

For datasets without unique IDs, the system uses:
1. Hash of question + answer (if available)
2. Hash of entire sample (fallback)

This ensures samples can still be matched even without explicit IDs.

## Default Behavior

**When no prior results exist**, the behavior is **exactly the same as before**:
- All samples are processed
- Results are saved normally
- No performance impact

The resume feature adds zero overhead when starting fresh.

## Benefits

1. **Time Savings**: Don't lose hours of work due to interruptions
2. **Resource Efficiency**: No need to re-process completed samples
3. **Flexibility**: Interrupt and resume at any time
4. **Robustness**: Handles various failure modes gracefully
5. **Transparency**: Clear logging of what's being skipped vs. processed

## Implementation Details

### Supported Datasets

Resume support is implemented for:
- All datasets using `BaseDataset` (PATH_VQA, VQA_RAD, SLAKE, etc.)
- MMMU validation and test sets
- Custom datasets (automatically supported if they inherit from BaseDataset)

### Sample Matching

Samples are matched using a unique key with stable hashing:
```python
import hashlib

def _get_sample_key(sample):
    if "id" in sample:
        return f"id:{sample['id']}"
    if "question" in sample and "answer" in sample:
        content = str(sample['question']) + str(sample['answer'])
        hash_value = hashlib.md5(content.encode()).hexdigest()
        return f"qa:{hash_value}"
    content = json.dumps(sample, sort_keys=True)
    hash_value = hashlib.md5(content.encode()).hexdigest()
    return f"hash:{hash_value}"
```

This ensures reliable matching across different dataset formats and Python interpreter restarts.

### Metrics Recalculation

Metrics are **always recalculated** on the complete dataset, ensuring:
- Correct accuracy even after resume
- Proper aggregation of statistics
- No stale metrics from partial runs

## Testing

Run the test suite to verify resume functionality:

```bash
python test_breakpoint_resume.py
```

This tests:
- Loading existing results
- Sample filtering and matching
- Single-chunk resume
- Multi-chunk resume
- Incremental total_results.json updates
- Edge case handling

## Troubleshooting

### "All samples already processed. Skipping inference."

This means all samples in the dataset have been completed. If you want to re-run:
1. Delete the results file: `rm eval_results/my_model/DATASET_NAME/results.json`
2. Re-run the evaluation command

### Metrics don't match expected values

Metrics are recalculated from scratch on the complete dataset. If they seem off:
1. Check that all result files are present and valid
2. Verify sample IDs are unique
3. Review the results.json file for any anomalies

### Resume not working

Ensure:
1. Output path is the same between runs
2. Dataset name matches exactly
3. Chunk indices match (for multi-chunk mode)
4. Result files are not corrupted (valid JSON)
