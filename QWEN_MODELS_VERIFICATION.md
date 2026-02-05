# Qwen Series Models - Chat Template Verification

## Issue Context
**Issue**: 未检查qwen系列模型  
**Translation**: Qwen series models not checked

## Purpose
Verify that all Qwen series models in the repository are using official chat templates via `apply_chat_template()`.

## Verification Summary

✅ **ALL QWEN MODELS ARE CORRECTLY IMPLEMENTED**

All 4 Qwen model variants (8 files total) are already using official chat templates through the processor's `apply_chat_template()` method. No changes are required.

## Models Verified

### 1. Qwen2_VL ✅
**Files**: 
- `models/Qwen2_VL/Qwen2_VL_hf.py`
- `models/Qwen2_VL/Qwen2_VL_vllm.py`

**Implementation**:
```python
# Line 35-39 in Qwen2_VL_hf.py
prompt = self.processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
```

**Status**: ✅ Using official chat template  
**Notes**: Properly handles image/images/text-only inputs

### 2. Qwen2.5_VL ✅
**Files**:
- `models/Qwen2_5_VL/Qwen2_5_VL_hf.py`
- `models/Qwen2_5_VL/Qwen2_5_VL_vllm.py`

**Implementation**:
```python
# Line 34-38 in Qwen2_5_VL_hf.py
prompt = self.processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
```

**Status**: ✅ Using official chat template  
**Notes**: Supports multi-turn conversations via "messages" format (vllm version)

### 3. Qwen3_VL ✅
**Files**:
- `models/Qwen3_VL/Qwen3_VL_hf.py`
- `models/Qwen3_VL/Qwen3_VL_vllm.py`

**Implementation**:
```python
# Line 45-50 in Qwen3_VL_hf.py
prompt = self.processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=self.enable_thinking,  # Qwen3 特性
)
```

**Status**: ✅ Using official chat template with Qwen3-specific feature  
**Special Feature**: `enable_thinking` parameter to control thinking mode (默认禁用，防止生成大量思维链)

### 4. Qwen2_VL_MoE ✅
**Files**:
- `models/Qwen2_VL_MoE/Qwen2_VL_hf.py`
- `models/Qwen2_VL_MoE/Qwen2_VL_vllm.py`

**Implementation**:
```python
# Line 37-41 in Qwen2_VL_MoE_hf.py
prompt = self.processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
```

**Status**: ✅ Using official chat template  
**Notes**: MoE (Mixture of Experts) variant, uses custom model class but same template approach

## Implementation Pattern

All Qwen models follow the same consistent pattern:

### 1. Message Construction
```python
new_messages = []
if "system" in messages:
    new_messages.append({"role": "system", "content": messages["system"]})

# Handle different input types: image, images, or text-only
if "image" in messages:
    new_messages.append({
        "role": "user",
        "content": [
            {"type": "image", "image": messages["image"]},
            {"type": "text", "text": messages["prompt"]}
        ]
    })
# ... (similar for images and text-only)
```

### 2. Apply Chat Template
```python
prompt = self.processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    # enable_thinking=... (Qwen3 only)
)
```

### 3. Process Vision Info
```python
image_inputs, video_inputs = process_vision_info(messages)
```

## Why Qwen Models Are Correct

1. **Official API**: Uses `processor.apply_chat_template()` which is the official HuggingFace method
2. **Proper Parameters**: 
   - `tokenize=False` - Returns string instead of token IDs
   - `add_generation_prompt=True` - Adds the assistant prompt for generation
3. **Vision Support**: Properly integrates with `qwen_vl_utils.process_vision_info()`
4. **Message Format**: Uses standard role-content message format compatible with OpenAI-style APIs

## Comparison with Fixed Models

### Before (Other Models - INCORRECT)
```python
# InternVL, BiMediX2, LLava_Med used custom templates
conv = conv_templates["mistral_instruct"]
conv.append_message(conv.roles[0], prompt)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()
```

### Qwen Models (CORRECT from Start)
```python
# Qwen models use official template
prompt = self.processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
```

## Benefits of Qwen Implementation

1. ✅ **Training-Format Alignment**: Prompts match exactly what model expects
2. ✅ **Special Tokens**: Correct BOS, EOS, and system tokens automatically handled
3. ✅ **Multi-turn Ready**: Supports conversation history naturally
4. ✅ **Maintainable**: Uses official API, less custom code
5. ✅ **Future-Proof**: Automatically benefits from processor updates

## Qwen3_VL Special Features

### Enable Thinking Mode
```python
# 默认禁用thinking模式，避免生成大量思维链
self.enable_thinking = getattr(args, 'enable_thinking', "False") == "True"

# 在apply_chat_template中使用
prompt = self.processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=self.enable_thinking
)
```

**Purpose**: Qwen3 默认开启thinking模式会生成大量思维链导致推理极慢。通过此参数可以禁用thinking模式以加快推理速度。

## Conclusion

✅ **ALL QWEN MODELS PASS VERIFICATION**

- **Total Files Verified**: 8
- **Using Official Templates**: 8/8 (100%)
- **Action Required**: None - all implementations are correct

The Qwen series models serve as excellent examples of proper chat template implementation and were not affected by the issues found in other model families (InternVL, BiMediX2, LLava_Med, HuatuoGPT).

## References

- [Qwen-VL Utils Documentation](https://github.com/QwenLM/Qwen-VL)
- [HuggingFace Chat Templates](https://huggingface.co/docs/transformers/chat_templating)
- [Qwen2-VL Model Card](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
- [Qwen3-VL Model Card](https://huggingface.co/Qwen/Qwen3-VL-7B-Instruct)

## Date
Verification completed: 2026-02-05
