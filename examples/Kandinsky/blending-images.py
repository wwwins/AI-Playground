"""
start->miniconda3
conda activate diffusers
cd Downloads
cd workspace
python blending-images.py
http://localhost:7860
"""
import torch
import numpy as np
import uuid
import os
import gradio as gr
from time import sleep
from random import randrange, randint
from datetime import datetime
from PIL import Image
from transformers import pipeline
from diffusers import (
    KandinskyPipeline,
    KandinskyPriorPipeline,
    KandinskyV22Pipeline,
    KandinskyV22PriorPipeline,
)
from diffusers.utils import load_image

# for windows
# from pathlib import Path
# CACHE_DIR = Path("c:/Users/User/.cache/huggingface/hub")
MODEL_DTYPE = torch.float16
LOW_MEMORY = True

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
torch.cuda.empty_cache()

##### version 2.2
model_prior = "kandinsky-community/kandinsky-2-2-prior"
model = "kandinsky-community/kandinsky-2-2-decoder"

pipe_prior = KandinskyV22PriorPipeline.from_pretrained(
    model_prior,
    torch_dtype=MODEL_DTYPE,
    low_cpu_mem_usage=LOW_MEMORY,
)
# pipe_prior.to("cpu")
pipe_prior.enable_model_cpu_offload()

pipe = KandinskyV22Pipeline.from_pretrained(
    model, torch_dtype=MODEL_DTYPE, low_cpu_mem_usage=LOW_MEMORY
)
# pipe.to("cuda")
pipe.enable_model_cpu_offload()
#####

MAX_SEED = torch.iinfo(torch.int32).max
# the same
# torch.manual_seed(0)
# generator = torch.Generator(device="cuda").manual_seed(0)
# multi GPU
# torch.cuda.manual_seed_all(0)

# prompt="monster cartoon style, unreal engine rendered, blend, close-mouth, small cute eyes, simplicity, clean background"
prompt = "photo of beauty woman sitting inside, full body, wearing a white t-shirt, white background"
negative_prompt = "bad, ugly, bad quality, low quality"
# negative_prompt="bad, ugly, low quality, watermark, logo, text, title, signature, words, letters, characters"
# negative_prompt="bad, ugly, lowres, error, worst quality, low quality, watermark, logo, text, title, signature, words, letters, characters, disfigured, immature, ugly eyes, cross-eye, over-saturated"
# negative_prompt ='lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature'

# depth_estimator = pipeline("depth-estimation")


def make_hint1(image, depth_estimator):
    depth = depth_estimator(image)["depth"]
    return depth


def make_hint(image, depth_estimator):
    image = depth_estimator(image)["depth"]
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    detected_map = torch.from_numpy(image).float() / 255.0
    depth = detected_map.permute(2, 0, 1)
    return depth


def get_depth_from_img(image):
    image = load_image(image)
    image = np.array(image)[:, :, 0]
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    detected_map = torch.from_numpy(image).float() / 255.0
    depth = detected_map.permute(2, 0, 1)
    return depth


def get_depth():
    # img = load_image("images/src.jpg")
    # hint = make_hint1(img, depth_estimator)
    # hint.save("images/depth.png")
    # hint = make_hint(img, depth_estimator).unsqueeze(0).half().to("cuda")
    hint = get_depth_from_img("images/depth.png")
    # print(hint)


def mixing_images(
    text_prompt,
    text_negative_prompt,
    text_weight,
    img1_weight,
    img2_weight,
    img3_weight,
    guidance,
    steps,
    seed,
    img1,
    img2,
    img3,
):
    print(
        "===============\nweight:{} {} {} {}\nguidance:{}\nsteps:{}\nseed:{}\n===============\n".format(
            text_weight, img1_weight, img2_weight, img3_weight, guidance, steps, seed
        )
    )
    # if img1.size != (1024,1024) or img2.size != (1024,1024):
    #    return None, "error"
    # img1 = Image.open("images/red.jpg")
    # img2 = Image.open("images/3eyes.jpg")
    # img_weight = (10-weight)*0.05
    # prompt_weight = weight * 0.1
    # images_prompt = [text_prompt, img1, img2]
    # weights = [prompt_weight, img_weight, img_weight]
    images_prompt = [text_prompt, img1, img2, img3]
    weights = [text_weight, img1_weight, img2_weight, img3_weight]
    if seed < 0:
        seed = randint(0, MAX_SEED)
    generator = torch.Generator(device="cuda").manual_seed(int(seed))
    image_emb, zero_image_emb = pipe_prior.interpolate(
        images_prompt,
        weights,
        # negative_prior_prompt = text_negative_prompt,
        # guidance_scale = guidance,
        # num_inference_steps = steps,
        # num_inference_steps=25,
        # num_images_per_prompt=1,
        generator=generator,
    ).to_tuple()
    # generator = torch.Generator(device="cuda").manual_seed(seed)
    out = pipe(
        image_embeds=image_emb,
        negative_image_embeds=zero_image_emb,
        height=768,
        width=768,
        guidance_scale=guidance,
        num_inference_steps=steps,
        generator=generator,
    ).images
    # s = randrange(20)
    # sleep(s)
    return out[0], "success:{}".format(seed)


def text2img():
    image_emb, negative_image_emb = pipe_prior(
        prompt, negative_prompt, guidance_scale=1.0
    ).to_tuple()
    out = pipe(
        prompt,
        image_embeds=image_emb,
        negative_image_embeds=negative_image_emb,
        height=512,
        width=512,
        num_inference_steps=100,
    ).images
    out[0].save("text2img.png")


def mixingv22():
    fileid = str(uuid.uuid1())
    fn = "{}.jpg".format(fileid)
    fullfn = "static\{}".format(fn)
    # img1 = Image.open("images/woman.jpg")
    # img2 = Image.open("images/ann.png")
    img1 = Image.open("images/monster-1.jpg")
    img2 = Image.open("images/deep-sea-1.png")
    images_prompt = [prompt, img1, img2]
    weights = [0.2, 0.4, 0.4]
    image_emb, zero_image_emb = pipe_prior.interpolate(
        images_prompt,
        weights,
    ).to_tuple()
    out = pipe(
        image_embeds=image_emb,
        negative_image_embeds=zero_image_emb,
        height=512,
        width=512,
    ).images
    #    prior_out = pipe_prior.interpolate(
    #            images_prompt,
    #            weights,
    #            negative_prior_prompt = negative_prompt,
    #            guidance_scale = 7
    #            )
    #    out = pipe(
    #            prompt,
    #            **prior_out,
    #            height=512,
    #            width=512,
    #            num_inference_steps = 100,
    #            ).images
    out[0].save(fullfn)


def mixing():
    img1 = Image.open("images/red.jpg")
    img2 = Image.open("images/3eyes.jpg")
    images_prompt = [prompt, img1, img2]
    weights = [0.2, 0.4, 0.4]
    # image_emb, negative_image_emb = pipe_prior.interpolate(images_prompt, weights, negative_prior_prompt=negative_prompt)
    # image_emb, negative_image_emb = pipe_prior.interpolate(images_prompt, weights)
    prior_out = pipe_prior.interpolate(
        images_prompt, weights, negative_prior_prompt=negative_prompt, guidance_scale=7
    )
    out = pipe(
        prompt,
        **prior_out,
        height=512,
        width=512,
        num_inference_steps=100,
    ).images
    # out = pipe(
    #        "",
    #        image_embeds = image_emb,
    #        negative_image_embeds = negative_image_emb,
    #        height = 512,
    #        width = 512,
    #        num_inference_steps = 150,
    #        ).images
    now = datetime.now()
    out[0].save("mixing-{}{}{}.png".format(now.day, now.hour, now.minute))


def check_gpu():
    print("Device name: {}".format(torch.cuda.get_device_name(0)))
    print("PyTorch version: {}".format(torch.__version__))
    print("CUDA version: {}".format(torch.version.cuda))
    print("cuDNN version: {}".format(torch.backends.cudnn.version()))


def demo():
    gr.Interface(
        fn=mixing_images,
        inputs=[
            gr.Textbox(value=prompt),
            gr.Textbox(value=negative_prompt),
            gr.Slider(
                0,
                1,
                value=0,
                step=0.1,
                label="weight of prompt",
                info="Choose between 0 and 1",
            ),
            gr.Slider(
                0,
                1,
                value=0.5,
                step=0.1,
                label="weight of img1",
                info="Choose between 0 and 1",
            ),
            gr.Slider(
                0,
                1,
                value=0.5,
                step=0.1,
                label="weight of img2",
                info="Choose between 0 and 1",
            ),
            gr.Slider(
                0,
                1,
                value=0.2,
                step=0.1,
                label="weight of img3",
                info="Choose between 0 and 1",
            ),
            gr.Slider(
                1,
                20,
                value=7,
                step=0.1,
                label="guidance scale",
                info="Choose between 1 and 20",
            ),
            gr.Slider(
                1,
                200,
                value=50,
                step=1,
                label="number inference steps",
                info="Choose between 1 and 200",
            ),
            gr.Number(value=-1),
            gr.Image(type="pil"),
            gr.Image(type="pil"),
            gr.Image(type="pil"),
        ],
        outputs=[gr.Image(label="mixing image"), gr.Textbox(label="result")],
    ).launch()


def main():
    check_gpu()
    # get_depth()
    # mixing()
    # mixingv22()
    # text2img()
    demo()


if __name__ == "__main__":
    main()
