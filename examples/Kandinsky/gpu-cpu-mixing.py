import sys
from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline
#from optimum.bettertransformer import BetterTransformer
import torch
import PIL
import os
import sys
import time
from diffusers.utils import load_image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection
from diffusers.models import UNet2DConditionModel
from uuid import uuid4
import numpy as np

DEVICE_CPU = torch.device('cpu:0')
DEVICE_GPU = torch.device('cuda:0')

# Loading encoder and prior pipeline into the RAM to be run on the CPU
# and unet and decoder to the VRAM to be run on the GPU.
# Note the usage of float32 for the CPU and float16 (half) for the GPU
# Set the `local_files_only` to True after the initial downloading
# to allow offline use (without active Internet connection)
print("*** Loading encoder ***")
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    'openai/clip-vit-base-patch32',
    # subfolder='image_encoder',
    # cache_dir='./kand22',
    # local_files_only=True
).to(DEVICE_CPU)

print("*** Loading unet ***")
# unet = UNet2DConditionModel.from_pretrained(
    # 'kandinsky-community/kandinsky-2-2-decoder',
    # subfolder='unet',
    # cache_dir='./kand22',
    # local_files_only=True
# ).half().to(DEVICE_GPU)

print("*** Loading prior ***")
prior = KandinskyV22PriorPipeline.from_pretrained(
    'kandinsky-community/kandinsky-2-2-prior',
    image_encoder=image_encoder, 
    torch_dtype=torch.float32,
    # cache_dir='./kand22',
    # local_files_only=True
).to(DEVICE_CPU)

print("*** Loading decoder ***")
decoder = KandinskyV22Pipeline.from_pretrained(
    'kandinsky-community/kandinsky-2-2-decoder',
    # unet=unet,
    torch_dtype=torch.float16,
    # cache_dir='./kand22',
    # local_files_only=True
).to(DEVICE_GPU)
#decoder = BetterTransformer.transform(decoder)

job_id = str(uuid4())

# torch.manual_seed(42)

num_batches = 1
images_per_batch = 1
total_num_images = images_per_batch * num_batches

prompt = 'photo of a beauty woman sitting inside, upper body, wearing a white t-shirt'
negative_prior_prompt = 'bad, ugly, bad quality, low quality, watermark, logo, text, title, signature, words, letters, characters, disfigured, immature, cartoon, anime, 3d, painting, ugly eyes, cross-eye, poorly drawn face, close-up, over-saturated colors, b&w, blurry, out of frame, cropped'


images = []

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
#start = time.time()
print(f"*** Generating {total_num_images} image(s) ***")
for i in range(num_batches):
    print(f"* Batch {i + 1} of {num_batches} *")
    
    # Generating embeddings on the CPU
    img_emb = prior(
        prompt=prompt,
        # num_inference_steps=25,
        num_images_per_prompt=images_per_batch)

    negative_emb = prior(
        prompt=negative_prior_prompt,
        # num_inference_steps=25,
        num_images_per_prompt=images_per_batch
    )

    # Converting fp32 to fp16, to run decoder on the GPU
    image_batch = decoder(
        image_embeds=img_emb.image_embeds.half(),
        negative_image_embeds=negative_emb.image_embeds.half(),
        # num_inference_steps=25,
        height=512,
        width=512)

    images += image_batch.images

end_event.record()
elapsed_time = start_event.elapsed_time(end_event)* 1.0e-3
max_gpu_memory = torch.cuda.max_memory_allocated(DEVICE_GPU)
print('Max GPU　memory:', max_gpu_memory*1e-9, ' GB')
print('Execution time:', elapsed_time/num_batches, 'seconds')
#end = time.time()
#print(f"total: {end - start}")
# 512x512: 57
# 1024x1024: 61
# Saving the images
os.mkdir(job_id)
for (idx, img) in enumerate(images):
    img.save(f"{job_id}/img_{job_id}_{idx + 1}.png")