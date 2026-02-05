from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import os
from PIL import Image

from .utils import DEFAULT_IMAGE_TOKEN,download

class LLavaMed:
    def __init__(self,model_path,args):
        super().__init__()
        self.llm = LLM(
            model= model_path,
            tensor_parallel_size= int(os.environ.get("tensor_parallel_size",1)),
            enforce_eager=True,
            limit_mm_per_prompt = {"image": int(os.environ.get("max_image_num",1))},
            # limit_mm_per_prompt = {"image":10}
        )
        
        # Load tokenizer for apply_chat_template
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self.sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            max_tokens= args.max_new_tokens,
            stop_token_ids=[],
        )

    def process_messages(self,messages):
        # Build messages in standard chat format for Mistral
        chat_messages = []
        
        # Add system message if present
        if "system" in messages:
            chat_messages.append({"role": "system", "content": messages["system"]})
        
        # Build user message content
        prompt_text = ""
        imgs = []
        
        if "image" in messages:
            text = messages["prompt"]
            prompt_text = DEFAULT_IMAGE_TOKEN + '\n' + text
            image = messages["image"]
            if isinstance(image, str):
                if os.path.exists(image):
                    image = Image.open(image)
            imgs.append(image)
        elif "images" in messages:
            text = messages["prompt"]
            images = messages["images"]
            prompt_parts = []
            for i, image in enumerate(images):
                prompt_parts.append(f"<image_{i+1}>: " + DEFAULT_IMAGE_TOKEN)
                if isinstance(image, str):
                    if os.path.exists(image):
                        image = Image.open(image)
                imgs.append(image)
            prompt_parts.append(text)
            prompt_text = '\n'.join(prompt_parts)
        else:
            prompt_text = messages["prompt"]
        
        chat_messages.append({"role": "user", "content": prompt_text})
        
        # Use tokenizer's apply_chat_template to format the prompt correctly
        # This ensures we use the official Mistral chat template
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
        # from pdb import set_trace;set_trace()
        outputs = self.llm.generate(llm_inputs_list, sampling_params=self.sampling_params)
        res = []
        for output in outputs:
            generated_text = output.outputs[0].text
            res.append(generated_text)
        return res
