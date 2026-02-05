import torch
from transformers import AutoModel, AutoTokenizer

from .utils import load_image


class InternVL:
    def __init__(self,model_path,args):
        super().__init__()
        self.llm =  AutoModel.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    device_map="cuda",
                    attn_implementation="flash_attention_2"
                    )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.generation_config ={ 
            'max_new_tokens': args.max_new_tokens,
            'repetition_penalty': args.repetition_penalty,
            'temperature' : args.temperature,
            'top_p' : args.top_p
        }
    

    def process_messages(self,messages):
        # Build messages in the standard chat format
        chat_messages = []
        
        # Add system message if present
        if "system" in messages:
            chat_messages.append({"role": "system", "content": messages["system"]})
        
        # Build user message content
        user_content = []
        pixel_value = None
        
        if "image" in messages:
            image = messages["image"]
            pixel_value = load_image(image).to(torch.bfloat16).to("cuda")
            user_content.append({"type": "image"})
            user_content.append({"type": "text", "text": messages["prompt"]})
        elif "images" in messages:
            images = messages["images"]
            for i, image in enumerate(images):
                user_content.append({"type": "text", "text": f"<image_{i+1}>: "})
                user_content.append({"type": "image"})
            user_content.append({"type": "text", "text": messages["prompt"]})
            pixel_value = [load_image(img).to(torch.bfloat16).to("cuda") for img in images]
            pixel_value = torch.cat(pixel_value, dim=0)
        else:
            user_content.append({"type": "text", "text": messages["prompt"]})
        
        chat_messages.append({"role": "user", "content": user_content})
        
        # Use tokenizer's apply_chat_template to format the prompt correctly
        # This ensures we use the official InternVL chat template
        prompt = self.tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        llm_inputs = {
            "prompt": prompt,
            "pixel_values": pixel_value
        }
        return llm_inputs


    def generate_output(self,messages):
        llm_inputs = self.process_messages(messages)
        question = llm_inputs["prompt"]
        pixel_values = llm_inputs["pixel_values"]
        response, history = self.llm.chat(self.tokenizer, pixel_values, question, self.generation_config,
                               history=None, return_history=True)
        return response
    
    def generate_outputs(self,messages_list):
        res = []
        for messages in messages_list:
            result = self.generate_output(messages)
            res.append(result)
        return res
