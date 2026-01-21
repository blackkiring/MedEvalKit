import os
from io import BytesIO
import requests
from PIL import Image
from tqdm import tqdm

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle, Conversation
from transformers import CLIPImageProcessor

import copy
import torch


def download_image(url):
    """Download image from URL and return as PIL Image."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/87.0.4280.88 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=120)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content)).convert('RGB')
        else:
            print(f"Failed to download image from URL: {url}")
            return None
    except Exception as e:
        print(f"Error downloading image from URL {url}: {e}")
        return None

# Register llava_llama_3 conversation template for BiMediX2 (Llama-3 based model)
if "llava_llama_3" not in conv_templates:
    # BiMediX2 uses Llama-3 format
    # Using TWO separator style which is compatible with Llama-3's conversation format
    conv_llava_llama_3 = Conversation(
        system="You are a helpful language and vision assistant. "
               "You are able to understand the visual content that the user provides, "
               "and assist the user with a variety of tasks using natural language.",
        roles=("user", "assistant"),
        version="llama_v3",
        messages=[],
        offset=0,
        sep_style=SeparatorStyle.TWO,
        sep="<|eot_id|>",
        sep2="<|eot_id|>",
    )
    conv_templates["llava_llama_3"] = conv_llava_llama_3

class BiMediX2:
    def __init__(self,model_path,args=None):
        super().__init__()
        # Use "llava" in model_name to trigger proper vision tower loading in LLaVA's builder
        # BiMediX2 is based on LLaVA architecture with Llama-3 backbone
        model_name = "llava_llama3_BiMediX2"
        device = "cuda"
        device_map = "auto"

        tokenizer, model, image_processor, max_length = load_pretrained_model(model_path, None, model_name, device_map=device_map)

        # Patch the model's forward method to handle newer transformers arguments
        # The old LLaVA code doesn't accept 'cache_position' which newer transformers passes
        self._patch_model_forward(model)

        # Set pad_token_id to suppress warning during generation
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.max_length = max_length
        self.conv_template = "llava_llama_3"

        self.temperature = args.temperature
        self.top_p = args.top_p
        self.repetition_penalty = args.repetition_penalty
        self.max_new_tokens = args.max_new_tokens
        '''
        self.temperature = 1
        self.top_p = 1
        self.repetition_penalty = 1.0
        self.max_new_tokens = 1024
        '''

    def _patch_model_forward(self, model):
        """Patch the model's forward method to accept newer transformers arguments."""
        original_forward = model.forward
        import functools

        @functools.wraps(original_forward)
        def patched_forward(*args, **kwargs):
            # Remove arguments that old LLaVA doesn't support
            kwargs.pop('cache_position', None)
            kwargs.pop('num_logits_to_keep', None)
            return original_forward(*args, **kwargs)

        model.forward = patched_forward

    def process_messages(self,messages):
        conv = copy.deepcopy(conv_templates[self.conv_template])
        conv.tokenizer = self.tokenizer
        conv.messages = []
        if  "system" in messages:
            conv.system = messages["system"]
        
        imgs = []
        if "image" in messages:
            image = messages["image"]
            if isinstance(image,str):
                image = Image.open(image).convert('RGB')
            else:
                image = image.convert('RGB')
            imgs.append(image)
            prompt = DEFAULT_IMAGE_TOKEN + '\n' + messages["prompt"]
        elif "images" in messages:
            images = messages["images"]
            prompt = ""
            for i,image in enumerate(images):
                prompt += f"<image_{i+1}>: " + DEFAULT_IMAGE_TOKEN + '\n'
                if isinstance(image,str):
                    if os.path.exists(image):
                        image = Image.open(image)
                    else:
                        image = download_image(image)
                elif isinstance(image,Image.Image):
                    image = image.convert("RGB")
                imgs.append(image)
            prompt += messages["prompt"]
        else:
            prompt = messages["prompt"]

        conv.append_message(conv.roles[0],prompt)
        conv.append_message(conv.roles[1],None) 
        prompt = conv.get_prompt()

        imgs = None if len(imgs) == 0 else imgs
        return prompt,imgs


    def generate_output(self,messages):
        prompt,imgs = self.process_messages(messages)
        if imgs:
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            imgs = process_images(imgs, self.image_processor, self.model.config)
            imgs = [_image.to(dtype=torch.float16, device="cuda") for _image in imgs]
        else:
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            imgs = None

        # Create attention_mask (all ones for non-padded input)
        attention_mask = torch.ones_like(input_ids)

        with torch.inference_mode():
            do_sample = False if self.temperature == 0 else True
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                images=imgs,
                do_sample = do_sample,
                temperature = self.temperature,
                top_p = self.top_p,
                repetition_penalty = self.repetition_penalty,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True)

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs
    
    def generate_outputs(self,messages_list):
        outputs = []
        for messages in tqdm(messages_list):
            output = self.generate_output(messages)
            outputs.append(output)
        return outputs
        