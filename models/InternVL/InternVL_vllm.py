from transformers import AutoProcessor
from vllm import LLM, SamplingParams
import os
from PIL import Image
import numpy as np

from .conversations import internvl_conv


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

        self.sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            max_tokens= args.max_new_tokens,
            stop_token_ids=[],
        )
    

    def process_messages(self,messages):
        conv = internvl_conv.copy()
        conv.messages = []
        imgs = []
        
        if "messages" in messages:
            messages = messages["messages"]
            for message in messages:
                role = message["role"]
                content = message["content"]
                if role == "user":
                    conv.append_message(conv.roles[0],content)
                else:
                    conv.append_message(conv.roles[1],content)

        else:
            if "system" in messages:
                conv.system_message = messages["system"]

            if "image" in messages:
                text = messages["prompt"]
                inp = "<image>" + '\n' + text
                conv.append_message(conv.roles[0],inp)
                image = messages["image"]
                image = ensure_pil_image(image)
                imgs.append(image)
            elif "images" in messages:
                text = messages["prompt"]
                images = messages["images"]
                inp = ""
                for i,image in enumerate(images):
                    inp = inp + f"<image_{i+1}>: " +"<image>" + '\n'
                    image = ensure_pil_image(image)
                    imgs.append(image)
                inp = inp + text
                conv.append_message(conv.roles[0],inp)
            else:
                text = messages["prompt"]
                inp = text
                conv.append_message(conv.roles[0],inp)
        
        conv.append_message(conv.roles[1],None) 
        prompt = conv.get_prompt()

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

