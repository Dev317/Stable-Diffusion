import os
from PIL import Image, ImageDraw
import cv2
import numpy as np
from base64 import b64encode

import torch
from torch import autocast
from torch.nn import functional as F
from diffusers import StableDiffusionPipeline, AutoencoderKL
from diffusers import UNet2DConditionModel, PNDMScheduler, LMSDiscreteScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, AutoProcessor, ASTModel
from tqdm.auto import tqdm

from IPython.display import HTML

import librosa

import platform
print(platform.platform())

device = 'cpu'
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
# else:
#     print ("MPS device not found.")

# STABLE_DIFFUSION_PIPELINE_PATH = './models/models--CompVis--stable-diffusion-v1-4/snapshots/2880f2ca379f41b0226444936bb7a6766a227587'
# pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4', torch_dtype=torch.float16, cache_dir=os.getenv("cache_dir", "./models"))
# pipe = StableDiffusionPipeline.from_pretrained(STABLE_DIFFUSION_PIPELINE_PATH, local_files_only=True)
# pipe = pipe.to(device)

# prompt = "An astronaut cat on a flying rocket to the moon, cyberpunk art"
# result = pipe(prompt)

# print(result.images)
# print(result.nsfw_content_detected)
# image = result.images[0]

# image.show()
# print(image)


# 1. Load the autoencoder model which will be used to decode the latents into image space.
# vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4', torch_dtype=torch.float32, subfolder='vae', cache_dir=os.getenv("cache_dir", "./models"))
VAE_PATH = './models/models--CompVis--stable-diffusion-v1-4/snapshots/249dd2d739844dea6a0bc7fc27b3c1d014720b28/vae'
vae = AutoencoderKL.from_pretrained(VAE_PATH, local_files_only=True)
vae = vae.to(device)

# 2. Load the tokenizer and text encoder to tokenize and encode the text.
# tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14', torch_dtype=torch.float32, cache_dir=os.getenv("cache_dir", "./models"))
TOKENIZER_PATH = './models/models--openai--clip-vit-large-patch14/snapshots/8d052a0f05efbaefbc9e8786ba291cfdf93e5bff'
tokenizer = CLIPTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)

# text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path='openai/clip-vit-large-patch14', torch_dtype=torch.float32, cache_dir=os.getenv("cache_dir", "./models"))
TEXT_ENCODER_PATH = './models/models--openai--clip-vit-large-patch14/snapshots/8d052a0f05efbaefbc9e8786ba291cfdf93e5bff'
text_encoder = CLIPTextModel.from_pretrained(TEXT_ENCODER_PATH, local_files_only=True)
text_encoder = text_encoder.to(device)

# 3. The UNet model for generating the latents.
# unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4', subfolder='unet', torch_dtype=torch.float32, cache_dir=os.getenv("cache_dir", "./models"))
UNET_PATH = './models/models--CompVis--stable-diffusion-v1-4/snapshots/249dd2d739844dea6a0bc7fc27b3c1d014720b28/unet'
unet = UNet2DConditionModel.from_pretrained(UNET_PATH, local_files_only=True)
unet = unet.to(device)

# 4. Create a scheduler for inference
# Handle denoising, makes things clean
scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule='scaled_linear', num_train_timesteps=1000)


AST_PATH = './models/models--MIT--ast-finetuned-audioset-10-10-0.4593/snapshots/c1c0c663ecf7a4de90db1bc2f8d4e2d38a4f93b4'
# auto_processor = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593",  cache_dir=os.getenv("cache_dir", "./models"))
auto_processor = AutoProcessor.from_pretrained(AST_PATH, local_files_only=True)

# ast = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", cache_dir=os.getenv("cache_dir", "./models"))
ast = ASTModel.from_pretrained(AST_PATH, local_files_only=True)


def get_text_embeds(prompt):

    # Tokenize the text and create embeddings
    text_input = tokenizer(text=prompt,
                            padding='max_length',
                            max_length=tokenizer.model_max_length,
                            truncation=True,
                            return_tensors='pt')

    print(text_input)
    print(text_input.input_ids.shape)

    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
        print(text_embeddings)
        print(text_embeddings.shape)

    # Unconditional embeddings, empty string extend to same length
    # For the purpose of classifier-free guidance
    unconditional_input = tokenizer(text=[''] * len(prompt),
                            padding='max_length',
                            max_length=tokenizer.model_max_length,
                            return_tensors='pt')

    # Concatenating text embeddings with unconditional embeddings
    with torch.no_grad():
        unconditional_embeddings = text_encoder(unconditional_input.input_ids.to(device))[0]

    final_text_embeddings = torch.cat([unconditional_embeddings, text_embeddings])
    print(final_text_embeddings)
    print(final_text_embeddings.shape)
    return final_text_embeddings

# test_embed = get_text_embeds(['a cute dog'])
# print(test_embed.shape)
# print(test_embed)

def produce_latents(embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None, return_all_latents=False):
    if latents is None:
        latents = torch.randn((embeddings.shape[0] // 2, unet.in_channels, height // 8, width // 8))
    latents = latents.to(device)

    latent_hist = [latents]

    scheduler.set_timesteps(num_inference_steps)
    latents = latents * scheduler.sigmas[0]

    for i, t in tqdm(enumerate(scheduler.timesteps)):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)
        sigma = scheduler.sigmas[i]
        latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=embeddings)['sample']

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents)['prev_sample']

        latent_hist.append(latents)

    if not return_all_latents:
        return latents

    all_latents = torch.cat(latent_hist, dim=0)
    return all_latents

def decode_img_latents(latents):
    latents = 1 / 0.18215 * latents

    with torch.no_grad():
        imgs = vae.decode(latents)['sample']

    imgs = (imgs / 2 + 0.5).clamp(0, 1)
    imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
    imgs = (imgs * 255).round().astype('uint8')
    pil_images = [Image.fromarray(image) for image in imgs]
    return pil_images


def prompt_to_img(prompts,
                  height=512, width=512,
                  num_inference_steps=50,
                  guidance_scale=7.5,
                  latents=None):

    if isinstance(prompts, str):
        prompts = [prompts]

    # Prompts -> text embeds
    text_embeddings = get_text_embeds(prompts)

    # Text embeds -> img latents
    latents = produce_latents(embeddings=text_embeddings,
                            height=height, width=width,
                            latents=latents,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale)

    # Img latents -> imgs
    imgs = decode_img_latents(latents)

    return imgs


def audio_to_img(filename,
                  height=512, width=512,
                  num_inference_steps=50,
                  guidance_scale=7.5,
                  latents=None,
                  return_all_latents=False,
                  batch_size=2):


    # audiofile -> audio embeds
    audio_embeddings = get_audio_embeds(filename)

    # audio embeds -> img latents
    latents = produce_latents(embeddings=audio_embeddings,
                            height=height, width=width,
                            latents=latents,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            return_all_latents=return_all_latents)

    # Img latents -> imgs
    # imgs = decode_img_latents(latents)
    # return imgs

    all_imgs = []
    for i in tqdm(range(0, len(latents), batch_size)):
        imgs = decode_img_latents(latents[i:i+batch_size])
        all_imgs.extend(imgs)

    return all_imgs



def get_audio_embeds(filename):

    y, sr = librosa.load(filename)

    # audio file is decoded on the fly
    inputs = auto_processor(y, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        outputs = ast(**inputs)

    last_hidden_layer = outputs.last_hidden_state
    final_embeddings = torch.cat([last_hidden_layer, last_hidden_layer])

    return final_embeddings


def imgs_to_video(imgs, video_name='video.mp4', fps=15):
    # Source: https://stackoverflow.com/questions/52414148/turn-pil-images-into-video-on-linux
    video_dims = (imgs[0].width, imgs[0].height)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter(video_name, fourcc, fps, video_dims)
    for img in imgs:
        tmp_img = img.copy()
        video.write(cv2.cvtColor(np.array(tmp_img), cv2.COLOR_RGB2BGR))
    video.release()

def display_video(file_path, width=512):
    os.system(f'ffmpeg -i {file_path} -vcodec libx264 {"./" + file_path}')
    mp4 = open(f'./{file_path}', 'rb').read()

    data_url = 'data:simul2/mp4;base64,' + b64encode(mp4).decode()
    return HTML("""
<video width={} controls>
    <source src="{}" type="video/mp4">
</video>
 """.format(width, data_url))



plt_img = prompt_to_img("", 512, 512, 20)[0]
print(plt_img)
plt_img.show()
get_text_embeds('a cute cat')

# plt_img = audio_to_img(filename='./bus_chatter.wav', height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None, return_all_latents=False, batch_size=2)[0]
# print(plt_img)
# plt_img.show()

# frames = audio_to_img('./bus_chatter.wav', num_inference_steps=20, return_all_latents=True)
# vid_name = 'sample.mp4'
# imgs_to_video(frames, vid_name)
# display_video(vid_name)