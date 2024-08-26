import spaces
import argparse
import os
import time
from os import path
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

cache_path = path.join(path.dirname(path.abspath(__file__)), "models")
os.environ["TRANSFORMERS_CACHE"] = cache_path
os.environ["HF_HUB_CACHE"] = cache_path
os.environ["HF_HOME"] = cache_path

import gradio as gr
import torch
from diffusers import FluxPipeline

torch.backends.cuda.matmul.allow_tf32 = True

class timer:
    def __init__(self, method_name="timed process"):
        self.method = method_name

    def __enter__(self):
        self.start = time.time()
        print(f"{self.method} starts")

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.time()
        print(f"{self.method} took {str(round(end - self.start, 2))}s")

if not path.exists(cache_path):
    os.makedirs(cache_path, exist_ok=True)

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.load_lora_weights(hf_hub_download("ByteDance/Hyper-SD", "Hyper-FLUX.1-dev-8steps-lora.safetensors"))
pipe.fuse_lora(lora_scale=0.125)
pipe.to(device="cuda", dtype=torch.bfloat16)

with gr.Blocks() as demo:
    with gr.Column():
        with gr.Row():
            with gr.Column():
                num_images = gr.Slider(label="Number of Images", minimum=1, maximum=8, step=1, value=4, interactive=True)
                height = gr.Number(label="Image Height", value=1024, interactive=True)
                width = gr.Number(label="Image Width", value=1024, interactive=True)
                # steps = gr.Slider(label="Inference Steps", minimum=1, maximum=8, step=1, value=1, interactive=True)
                # eta = gr.Number(label="Eta (Corresponds to parameter eta (η) in the DDIM paper, i.e. 0.0 eqauls DDIM, 1.0 equals LCM)", value=1., interactive=True)
                prompt = gr.Text(label="Prompt", value="a photo of a cat", interactive=True)
                seed = gr.Number(label="Seed", value=3413, interactive=True)
                btn = gr.Button(value="run")
            with gr.Column():
                output = gr.Gallery(height=1024)

            @spaces.GPU
            def process_image(num_images, height, width, prompt, seed):
                global pipe
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16), timer("inference"):
                    return pipe(
                        prompt=[prompt]*num_images,
                        generator=torch.Generator().manual_seed(int(seed)),
                        num_inference_steps=8,
                        guidance_scale=3.5,
                        height=int(height),
                        width=int(width)
                    ).images

            reactive_controls = [num_images, height, width, prompt, seed]

            # for control in reactive_controls:
            #     control.change(fn=process_image, inputs=reactive_controls, outputs=[output])

            btn.click(process_image, inputs=reactive_controls, outputs=[output])

if __name__ == "__main__":
    demo.launch()