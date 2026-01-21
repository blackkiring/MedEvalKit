from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import torch
from tqdm import tqdm
import tempfile
import os


class Hulu_Med:
    def __init__(self, model_path, args=None):
        super().__init__()

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        self.tokenizer = self.processor.tokenizer

        self.temperature = args.temperature
        self.top_p = args.top_p
        self.repetition_penalty = args.repetition_penalty
        self.max_new_tokens = args.max_new_tokens

        # Temp directory for saving PIL images
        self.temp_dir = tempfile.mkdtemp()

    def _save_image_if_needed(self, image, idx=0):
        """If image is PIL Image, save to temp file and return path. Otherwise return as-is."""
        if isinstance(image, str):
            return image
        elif isinstance(image, Image.Image):
            temp_path = os.path.join(self.temp_dir, f"temp_image_{idx}.jpg")
            image.save(temp_path)
            return temp_path
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def process_messages(self, messages):
        """按官方格式构建 conversation"""
        conversation = []

        # System prompt
        if "system" in messages:
            conversation.append({
                "role": "system",
                "content": messages["system"]
            })

        # User message
        content = []
        modal = "text"  # 默认文本模态

        if "image" in messages:
            # 单图像：保存为临时文件并使用 image_path 格式
            image_path = self._save_image_if_needed(messages["image"], 0)
            content.append({
                "type": "image",
                "image": {"image_path": image_path}
            })
            modal = "image"
        elif "images" in messages:
            # 多图像
            for i, img in enumerate(messages["images"]):
                content.append({"type": "text", "text": f"Image {i+1}: "})
                image_path = self._save_image_if_needed(img, i)
                content.append({
                    "type": "image",
                    "image": {"image_path": image_path}
                })
            modal = "image"

        content.append({"type": "text", "text": messages["prompt"]})
        conversation.append({"role": "user", "content": content})

        # 使用官方 processor 调用方式
        inputs = self.processor(
            conversation=conversation,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        # 转换到 GPU 和正确的 dtype
        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        return inputs, modal

    def generate_output(self, messages):
        inputs, modal = self.process_messages(messages)

        do_sample = self.temperature > 0

        # Remove modals from inputs if already present to avoid duplicate
        if "modals" in inputs:
            inputs.pop("modals")

        generated_ids = self.model.generate(
            **inputs,
            modals=[modal],
            do_sample=do_sample,
            temperature=self.temperature if do_sample else None,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            max_new_tokens=self.max_new_tokens,
            use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        output_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            use_think=False
        )[0].strip()

        return output_text

    def generate_outputs(self, messages_list):
        outputs = []
        for messages in tqdm(messages_list):
            output = self.generate_output(messages)
            outputs.append(output)
        return outputs
