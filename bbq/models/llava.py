import requests
from io import BytesIO

import torch
torch.autograd.set_grad_enabled(False)
from PIL import Image
from transformers import TextStreamer
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()
from llava.utils import disable_torch_init
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

class LLaVaChat:
    # Model Constants
    IGNORE_INDEX = -100
    IMAGE_TOKEN_INDEX = -200
    DEFAULT_IMAGE_TOKEN = "<image>"
    DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
    DEFAULT_IM_START_TOKEN = "<im_start>"
    DEFAULT_IM_END_TOKEN = "<im_end>"
    IMAGE_PLACEHOLDER = "<image-placeholder>"

    def __init__(self, model_path="liuhaotian/llava-v1.6-vicuna-7b"):
        disable_torch_init()

        self.model_name = get_model_name_from_path(model_path)  
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
        model_path, None, self.model_name, device="cuda")

        if "llama-2" in self.model_name.lower():
            self.conv_mode = "llava_llama_2"
        elif "mistral" in self.model_name.lower():
            self.conv_mode = "mistral_instruct"
        elif "v1.6-34b" in self.model_name.lower():
            self.conv_mode = "chatml_direct"
        elif "v1" in self.model_name.lower():
            self.conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            self.conv_mode = "mpt"
        else:
            self.conv_mode = "llava_v0"

    def preprocess_image(self, images):
        x = process_images(
            images,
            self.image_processor,
            self.model.config)

        return x.to(self.model.device, dtype=torch.float16)

    def __call__(self, query, image_features, image_sizes):
        # Given this query, and the image_featurese, prompt LLaVA with the query,
        # using the image_features as context.

        conv = conv_templates[self.conv_mode].copy()

        if self.model.config.mm_use_im_start_end:
            inp = LLaVaChat.DEFAULT_IM_START_TOKEN +\
                  LLaVaChat.DEFAULT_IMAGE_TOKEN +\
                  LLaVaChat.DEFAULT_IM_END_TOKEN + '\n' + query
        else:
            inp = LLaVaChat.DEFAULT_IMAGE_TOKEN + '\n' + query
        conv.append_message(conv.roles[0], inp)

        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, LLaVaChat.IMAGE_TOKEN_INDEX,
            return_tensors='pt').unsqueeze(0).to("cuda")
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        self.temperature = 0
        self.max_new_tokens = 512
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_features,
                image_sizes=image_sizes,
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                streamer=streamer,
                use_cache=True)

        outputs = self.tokenizer.decode(output_ids[0]).strip()
        return outputs

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

