import os
import torch
from diffusers import AutoencoderKL
from diffusers import UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4',
                                    torch_dtype=torch.float32, subfolder='vae', cache_dir=os.getenv("cache_dir", "./models"))

tokenizer = CLIPTokenizer.from_pretrained(
    'openai/clip-vit-large-patch14', torch_dtype=torch.float32, cache_dir=os.getenv("cache_dir", "./models"))

text_encoder = CLIPTextModel.from_pretrained(
    pretrained_model_name_or_path='openai/clip-vit-large-patch14', torch_dtype=torch.float32, cache_dir=os.getenv("cache_dir", "./models"))

unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4',
                                            subfolder='unet', torch_dtype=torch.float32, cache_dir=os.getenv("cache_dir", "./models"))
