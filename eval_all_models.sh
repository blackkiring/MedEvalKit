#!/bin/bash
# 批量评估所有模型的脚本
# Usage: ./eval_all_models.sh [dataset] [use_vllm]
# Example: ./eval_all_models.sh "VQA_RAD" "False"
# Example: ./eval_all_models.sh "VQA_RAD,SLAKE,PATH_VQA" "True"

export HF_ENDPOINT=https://hf-mirror.com

# 可通过命令行参数覆盖
# EVAL_DATASETS="${1:-VQA_RAD}"
EVAL_DATASETS="MMMU-Medical-test,VQA_RAD,SLAKE,PATH_VQA,PMC_VQA,OmniMedVQA,MedXpertQA-MM,MMLU,PubMedQA,MedMCQA,MedQA_USMLE,Medbullets_op4,Medbullets_op5,MedXpertQA-Text,SuperGPQA,MIMIC_CXR,IU_XRAY"
USE_VLLM="${2:-False}"

DATASETS_PATH="hf"

# 定义所有模型配置 (MODEL_NAME|MODEL_PATH)
MODELS=(
    # Qwen 系列
    "Qwen2-VL|JZPeterPan/MedVLM-R1"
    "Qwen2.5-VL|Qwen/Qwen2.5-VL-3B-Instruct"
    "Qwen2.5-VL|Qwen/Qwen2.5-VL-7B-Instruct"
    "Qwen3-VL|Qwen/Qwen3-VL-2B-Instruct"
    "Qwen3-VL|Qwen/Qwen3-VL-4B-Instruct"
    "Qwen3-VL|Qwen/Qwen3-VL-8B-Instruct"

    # InternVL 系列
    "InternVL|OpenGVLab/InternVL2_5-8B"
    "InternVL|OpenGVLab/InternVL3-8B"
    "InternVL|OpenGVLab/InternVL3_5-8B"

    # 医疗专用模型
    "MedGemma|google/medgemma-4b-it"
    "LLava_Med|microsoft/llava-med-v1.5-mistral-7b"
    "Huatuo|FreedomIntelligence/HuatuoGPT-Vision-7B"
    "BiMediX2|MBZUAI/BiMediX2-8B"
    "Lingshu|lingshu-medical-mllm/Lingshu-7B"
    "Hulu-Med|ZJU-AI4H/Hulu-Med-7B"
)

# GPU 配置
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPULIST <<< "$CUDA_VISIBLE_DEVICES"
TOTAL_GPUS=${#GPULIST[@]}
CHUNKS=${CHUNKS:-$TOTAL_GPUS}

# 评估参数
SEED=42
REASONING="False"
TEST_TIMES=1

# 模型生成参数
MAX_NEW_TOKENS=1024
MAX_IMAGE_NUM=6
TEMPERATURE=0
TOP_P=0.0001
REPETITION_PENALTY=1
ENABLE_THINKING="False"

# LLM Judge 配置
USE_LLM_JUDGE="False"
GPT_MODEL="kimi-k2-turbo-preview"
JUDGE_MODEL_TYPE="openai"
API_KEY="sk-aZpGiwxPi27MtsXzRwxRWeYVTJsP8JUnlgv8TPEFoIvXYRFQ"
BASE_URL="https://api.moonshot.cn/v1"

# 日志目录
LOG_DIR="eval_logs"
mkdir -p "$LOG_DIR"

# 记录开始时间
START_TIME=$(date +%s)
echo "=========================================="
echo "开始批量评估 - $(date)"
echo "评估数据集: $EVAL_DATASETS"
echo "使用 vLLM: $USE_VLLM"
echo "模型数量: ${#MODELS[@]}"
echo "=========================================="

# 评估每个模型
for MODEL_CONFIG in "${MODELS[@]}"; do
    # 解析 MODEL_NAME 和 MODEL_PATH
    IFS='|' read -r MODEL_NAME MODEL_PATH <<< "$MODEL_CONFIG"

    OUTPUT_PATH="eval_results/${MODEL_PATH}"
    LOG_FILE="${LOG_DIR}/${MODEL_PATH//\//_}.log"

    echo ""
    echo "----------------------------------------"
    echo "[$(date +%H:%M:%S)] 开始评估: $MODEL_NAME ($MODEL_PATH)"
    echo "输出路径: $OUTPUT_PATH"
    echo "日志文件: $LOG_FILE"
    echo "----------------------------------------"

    MODEL_START_TIME=$(date +%s)

    # 并行运行多个 chunk
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} uv run python eval.py \
            --eval_datasets "$EVAL_DATASETS" \
            --datasets_path "$DATASETS_PATH" \
            --output_path "$OUTPUT_PATH" \
            --model_name "$MODEL_NAME" \
            --model_path "$MODEL_PATH" \
            --seed $SEED \
            --max_new_tokens "$MAX_NEW_TOKENS" \
            --max_image_num "$MAX_IMAGE_NUM" \
            --use_vllm "$USE_VLLM" \
            --num_chunks $CHUNKS \
            --chunk_idx $IDX \
            --reasoning $REASONING \
            --temperature "$TEMPERATURE" \
            --top_p "$TOP_P" \
            --repetition_penalty "$REPETITION_PENALTY" \
            --enable_thinking "$ENABLE_THINKING" \
            --use_llm_judge "$USE_LLM_JUDGE" \
            --judge_model_type "$JUDGE_MODEL_TYPE" \
            --judge_model "$GPT_MODEL" \
            --api_key "$API_KEY" \
            --base_url "$BASE_URL" \
            --test_times "$TEST_TIMES" \
            >> "$LOG_FILE" 2>&1 &
    done

    # 等待当前模型的所有 chunk 完成
    wait

    MODEL_END_TIME=$(date +%s)
    MODEL_DURATION=$((MODEL_END_TIME - MODEL_START_TIME))
    echo "[$(date +%H:%M:%S)] 完成评估: $MODEL_NAME (耗时: ${MODEL_DURATION}秒)"

    # 检查是否成功
    if grep -q "final results" "$LOG_FILE" 2>/dev/null; then
        echo "✓ 评估成功"
        # 显示最终结果
        grep "final results" "$LOG_FILE" | tail -1
    else
        echo "✗ 评估可能失败，请检查日志: $LOG_FILE"
    fi
done

# 总结
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
echo ""
echo "=========================================="
echo "批量评估完成 - $(date)"
echo "总耗时: ${TOTAL_DURATION}秒 ($((TOTAL_DURATION/60))分钟)"
echo "日志保存在: $LOG_DIR/"
echo "=========================================="

# 汇总所有结果
echo ""
echo "评估结果汇总:"
echo "----------------------------------------"
for MODEL_CONFIG in "${MODELS[@]}"; do
    IFS='|' read -r MODEL_NAME MODEL_PATH <<< "$MODEL_CONFIG"
    LOG_FILE="${LOG_DIR}/${MODEL_PATH//\//_}.log"

    if [ -f "$LOG_FILE" ]; then
        RESULT=$(grep "final results" "$LOG_FILE" 2>/dev/null | tail -1)
        if [ -n "$RESULT" ]; then
            echo "[$MODEL_NAME] $MODEL_PATH:"
            echo "  $RESULT"
        else
            echo "[$MODEL_NAME] $MODEL_PATH: 无结果 (检查 $LOG_FILE)"
        fi
    fi
done
