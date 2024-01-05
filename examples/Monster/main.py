"""
start->miniconda3
conda activate diffusers
cd Downloads
cd workspace
python main.py
http://localhost:7860
"""
import torch
import numpy as np
import os
import gradio as gr
from pathlib import Path
from PIL import Image
from transformers import pipeline
from monster_diffusion import MonsterDiffusion

# for windows
# CACHE_DIR = Path("c:/Users/User/.cache/huggingface/hub")
# m = MonsterDiffusion(cache_dir=CACHE_DIR)
m = MonsterDiffusion()


def mixing_images(
    img1,
    img2,
    img3,
    img1_weight,
    img2_weight,
    img3_weight,
    guidance,
    steps,
    seed,
):
    print(
        "===============\nweight:{} {} {}\nguidance:{}\nsteps:{}\nseed:{}\n===============\n".format(
            img1_weight, img2_weight, img3_weight, guidance, steps, seed
        )
    )
    if seed < 0:
        seed = None
    out, seed = m.mixing(img1, img2, seed=seed)
    return out[0], "success:{}".format(seed)


def check_gpu():
    print("Device name: {}".format(torch.cuda.get_device_name(0)))
    print("PyTorch version: {}".format(torch.__version__))
    print("CUDA version: {}".format(torch.version.cuda))
    print("cuDNN version: {}".format(torch.backends.cudnn.version()))


def demo():
    gr.Interface(
        fn=mixing_images,
        inputs=[
            gr.Image(type="pil"),
            gr.Image(type="pil"),
            gr.Image(type="pil"),
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
        ],
        outputs=[gr.Image(label="mixing image"), gr.Textbox(label="result")],
    ).launch()


def main():
    check_gpu()
    demo()


if __name__ == "__main__":
    main()
