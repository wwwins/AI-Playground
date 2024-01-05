import torch
import numpy as np
from PIL import Image
from random import randrange, randint
from transformers import pipeline
from diffusers.utils import load_image
from diffusers import KandinskyV22Pipeline as AutoPipeline
from diffusers import KandinskyV22PriorPipeline as AutoPriorPipeline


class MonsterDiffusion:
    """
    from monster_diffusion import MonsterDiffusion
    m = MonsterDiffusion()
    m.mixing(img1, img2)
    or
    CACHE_DIR = Path("c:/Users/User/.cache/huggingface/hub")
    m = MonsterDiffusion(cache_dir=CACHE_DIR)
    m.mixing(img1, img2, seed=seed)

    """

    def __init__(
        self,
        model_prior="kandinsky-community/kandinsky-2-2-prior",
        model="kandinsky-community/kandinsky-2-2-decoder",
        model_dtype=torch.float16,
        low_memory=True,
        cache_dir=None,
    ):
        try:
            torch.cuda.empty_cache()
            self.pipe_prior = AutoPriorPipeline.from_pretrained(
                model_prior,
                torch_dtype=model_dtype,
                low_cpu_mem_usage=low_memory,
                cache_dir=cache_dir,
            )
            self.pipe_prior.enable_model_cpu_offload()

            self.pipe = AutoPipeline.from_pretrained(
                model,
                torch_dtype=model_dtype,
                low_cpu_mem_usage=low_memory,
                cache_dir=cache_dir,
            )
            self.pipe.enable_model_cpu_offload()
        except Exception as e:
            print("Error:", e)

    def mixing(self, img1, img2, prompt="monster", seed=None):
        fixed = Image.new("RGB", (1024, 1024), color=(224, 224, 224))
        images_prompt = [prompt, img1, img2, fixed]
        if seed is None:
            seed = torch.randint(0, 2**63 - 1, []).item()
        generator = torch.Generator(device="cuda").manual_seed(int(seed))

        image_emb, zero_image_emb = self.pipe_prior.interpolate(
            images_prompt,
            [0, 0.5, 0.5, 0.2],
            generator=generator,
        ).to_tuple()

        out = self.pipe(
            image_embeds=image_emb,
            negative_image_embeds=zero_image_emb,
            height=768,
            width=768,
            guidance_scale=7,
            num_inference_steps=50,
            generator=generator,
        ).images
        return (out, seed)
