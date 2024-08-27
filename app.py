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

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        <div style="text-align: center; max-width: 650px; margin: 0 auto;">
            <h1 style="font-size: 2.5rem; font-weight: 700; margin-bottom: 1rem;">FLUX Image Generator</h1>
            <p style="font-size: 1rem; margin-bottom: 1.5rem;">Create unique images with AI. Just describe what you want to see!</p>
        </div>
        """
    )
    
    with gr.Group():
        prompt = gr.Textbox(
            label="Your Image Description",
            placeholder="E.g., A serene landscape with mountains and a lake at sunset",
            lines=3
        )
        
        with gr.Accordion("Advanced Settings", open=False):
            with gr.Group():
                with gr.Row():
                    height = gr.Slider(label="Height", minimum=256, maximum=1024, step=64, value=1024)
                    width = gr.Slider(label="Width", minimum=256, maximum=1024, step=64, value=1024)
                
                with gr.Row():
                    steps = gr.Slider(label="Inference Steps", minimum=6, maximum=25, step=1, value=8)
                    scales = gr.Slider(label="Guidance Scale", minimum=0.0, maximum=5.0, step=0.1, value=3.5)
                
                seed = gr.Number(label="Seed (for reproducibility)", value=3413, precision=0)
        
        generate_btn = gr.Button("Generate Image", variant="primary", scale=1)
        
    output = gr.Image(label="Your Generated Image")
    
    gr.Markdown(
        """
        <div style="max-width: 650px; margin: 2rem auto; padding: 1rem; border-radius: 10px; background-color: #f0f0f0;">
            <h2 style="font-size: 1.5rem; margin-bottom: 1rem;">How to Use</h2>
            <ol style="padding-left: 1.5rem;">
                <li>Enter a detailed description of the image you want to create.</li>
                <li>Adjust advanced settings if desired (tap to expand).</li>
                <li>Tap "Generate Image" and wait for your creation!</li>
            </ol>
            <p style="margin-top: 1rem; font-style: italic;">Tip: Be specific in your description for best results!</p>
        </div>
        """
    )

    @spaces.GPU
    def process_image(height, width, steps, scales, prompt, seed):
        global pipe
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16), timer("inference"):
            return pipe(
                prompt=[prompt],
                generator=torch.Generator().manual_seed(int(seed)),
                num_inference_steps=int(steps),
                guidance_scale=float(scales),
                height=int(height),
                width=int(width)
            ).images

    generate_btn.click(
        process_image,
        inputs=[height, width, steps, scales, prompt, seed],
        outputs=[output]
    )

if __name__ == "__main__":
    demo.launch()
