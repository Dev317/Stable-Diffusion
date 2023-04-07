from PIL import Image

import torch
from diffusers import AutoencoderKL
from diffusers import UNet2DConditionModel, LMSDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm

import librosa
import numpy as np

device = 'cpu'

# 1. Load the autoencoder model which will be used to decode the latents into image space.
VAE_PATH = './models/models--CompVis--stable-diffusion-v1-4/snapshots/249dd2d739844dea6a0bc7fc27b3c1d014720b28/vae'
vae = AutoencoderKL.from_pretrained(VAE_PATH, local_files_only=True)
vae = vae.to(device)

# 2. Load the tokenizer and text encoder to tokenize and encode the text.
TOKENIZER_PATH = './models/models--openai--clip-vit-large-patch14/snapshots/8d052a0f05efbaefbc9e8786ba291cfdf93e5bff'
tokenizer = CLIPTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)

TEXT_ENCODER_PATH = './models/models--openai--clip-vit-large-patch14/snapshots/8d052a0f05efbaefbc9e8786ba291cfdf93e5bff'
text_encoder = CLIPTextModel.from_pretrained(TEXT_ENCODER_PATH, local_files_only=True)
text_encoder = text_encoder.to(device)

# 3. The UNet model for generating the latents.
UNET_PATH = './models/models--CompVis--stable-diffusion-v1-4/snapshots/249dd2d739844dea6a0bc7fc27b3c1d014720b28/unet'
unet = UNet2DConditionModel.from_pretrained(UNET_PATH, local_files_only=True)
unet = unet.to(device)

# 4. Create a scheduler for inference
# Handle denoising, makes things clean
scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule='scaled_linear', num_train_timesteps=1000)


def get_text_embeds(prompt):

    # Tokenize the text and create embeddings
    text_input = tokenizer(text=prompt,
                            padding='max_length',
                            max_length=tokenizer.model_max_length,
                            truncation=True,
                            return_tensors='pt')

    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

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
    return final_text_embeddings


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


max_pad_len = 174
num_rows = 40
num_columns = 174
num_channels = 1

classes = {
    0 : "air conditioner",
    1 : "car horn",
    2 : "children playing",
    3 : "dog bark",
    4 : "drilling",
    5 : "engine idling",
    6 : "gun shot",
    7 : "jackhammer",
    8 : "siren",
    9 : "street music",
}

def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]

        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = np.delete(mfccs, np.s_[max_pad_len::1], 1)

    except Exception as e:
        print(str(e))
        return None

    return mfccs


def predict(file_name, model):
    prediction_feature = extract_features(file_name)
    prediction_feature = prediction_feature.reshape(1, num_rows, num_columns, num_channels)

    predicted_vector = model.predict(prediction_feature)
    predicted_class = np.argmax(predicted_vector)

    return classes[predicted_class]
