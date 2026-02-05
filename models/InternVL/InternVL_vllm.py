from transformers import AutoProcessor
from vllm import LLM, SamplingParams
import os
from PIL import Image
import numpy as np


def ensure_pil_image(image):
    """Ensure image is a proper PIL Image with RGB mode and uint8 dtype."""
    if isinstance(image, str):
        if os.path.exists(image):
            image = Image.open(image).convert("RGB")
        else:
            raise FileNotFoundError(f"Image not found: {image}")
    elif isinstance(image, Image.Image):
        image = image.convert("RGB")
    elif isinstance(image, np.ndarray):
        # Handle numpy arrays - convert to uint8 if needed
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        image = Image.fromarray(image).convert("RGB")
    return image


class InternVL:
    def __init__(self,model_path,args):
        super().__init__()
        self.llm = LLM(
            model= model_path,
            trust_remote_code=True,
            tensor_parallel_size= int(os.environ.get("tensor_parallel_size",4)),
            enforce_eager=True,
            gpu_memory_utilization = 0.7,
            limit_mm_per_prompt = {"image": int(os.environ.get("max_image_num",1))},
            dtype="bfloat16",  # Explicitly set dtype
            max_model_len=8192,  # Limit context length
        )
        self.processor = AutoProcessor.from_pretrained(model_path,trust_remote_code=True)
        # Get tokenizer from processor for apply_chat_template
        self.tokenizer = self.processor.tokenizer

        self.sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            max_tokens= args.max_new_tokens,
            stop_token_ids=[],
        )
    

    def process_messages(self,messages):
        # Build messages in standard chat format
        chat_messages = []
        imgs = []
        
        if "messages" in messages:
            # Chat history format - convert to standard format
            messages_list = messages["messages"]
            for message in messages_list:
                role = message["role"]
                content = message["content"]
                # Keep the content as-is, it may already be formatted
                chat_messages.append({"role": role, "content": content})
        else:
            # Single prompt format
            if "system" in messages:
                chat_messages.append({"role": "system", "content": messages["system"]})
            
            # Build user message content
            user_content = []
            if "image" in messages:
                image = ensure_pil_image(messages["image"])
                imgs.append(image)
                user_content.append({"type": "image"})
                user_content.append({"type": "text", "text": messages["prompt"]})
            elif "images" in messages:
                images = messages["images"]
                for i, img in enumerate(images):
                    user_content.append({"type": "text", "text": f"<image_{i+1}>: "})
                    image = ensure_pil_image(img)
                    imgs.append(image)
                    user_content.append({"type": "image"})
                user_content.append({"type": "text", "text": messages["prompt"]})
            else:
                user_content = messages["prompt"]
            
            chat_messages.append({"role": "user", "content": user_content})
        
        # Use tokenizer's apply_chat_template to format the prompt correctly
        # This ensures we use the official InternVL chat template
        prompt = self.tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True
        )

        mm_data = {}
        if len(imgs) > 0:
            mm_data["image"] = imgs

        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
        }

        return llm_inputs


    def generate_output(self,messages):
        llm_inputs = self.process_messages(messages)
        outputs = self.llm.generate([llm_inputs], sampling_params=self.sampling_params)
        return outputs[0].outputs[0].text
    
    def generate_outputs(self,messages_list):
        llm_inputs_list = [self.process_messages(messages) for messages in messages_list]
        outputs = self.llm.generate(llm_inputs_list, sampling_params=self.sampling_params)
        res = []
        for output in outputs:
            generated_text = output.outputs[0].text
            res.append(generated_text)
        return res

