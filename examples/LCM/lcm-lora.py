from diffusers import DiffusionPipeline, LCMScheduler
from random import randrange, randint
import torch
import gradio as gr

PROMPT = "photo of pizza, leica 35mm summilux"
# PROMPT = "close-up photography of old man standing in the rain at night, in a street lit by lamps, leica 35mm summilux"
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
LCM_LORA_ID = "latent-consistency/lcm-lora-sdxl"
MAX_SEED = torch.iinfo(torch.int32).max

if torch.cuda.is_available():
    TORCH_DEVICE = torch.device("cuda")
    TORCH_DTYPE = torch.float16
else:
    TORCH_DEVICE = torch.device("cpu")
    TORCH_DTYPE = torch.float32

pipe = DiffusionPipeline.from_pretrained(MODEL_ID, variant="fp16")
pipe.load_lora_weights(LCM_LORA_ID)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.to(device=TORCH_DEVICE, dtype=TORCH_DTYPE)


def text2img(text_prompt, num_inference_steps, guidance_scale, seed):
    if seed < 0:
        seed = randint(0, MAX_SEED)
    generator = torch.Generator(device=pipe.device).manual_seed(int(seed))
    output = pipe(
        prompt=text_prompt,
        width=1024,
        height=1024,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images
    return output[0], "success:{} {}".format(text_prompt, seed)


def demo():
    gr.Interface(
        fn=text2img,
        inputs=[
            gr.Textbox(value=PROMPT),
            gr.Slider(
                1,
                20,
                value=4,
                step=1,
                label="number inference steps",
                info="Choose between 1 and 20",
            ),
            gr.Slider(
                1,
                2,
                value=1,
                step=0.1,
                label="guidance scale",
                info="Choose between 1 and 2",
            ),
            gr.Number(value=-1),
        ],
        outputs=[gr.Image(label="mixing image"), gr.Textbox(label="result")],
    ).launch()


def check_gpu():
    print("Device name: {}".format(torch.cuda.get_device_name(0)))
    print("PyTorch version: {}".format(torch.__version__))
    print("CUDA version: {}".format(torch.version.cuda))
    print("cuDNN version: {}".format(torch.backends.cudnn.version()))


def main():
    if torch.cuda.is_available():
        check_gpu()
    demo()


if __name__ == "__main__":
    main()
