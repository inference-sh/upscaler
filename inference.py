from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
import torch
from PIL import Image
from .upscale import upscale, UpscaleMode, SeamFixMode
from typing import Optional
from io import BytesIO
from diffusers import FluxInpaintPipeline
from RealESRGAN import RealESRGAN
from huggingface_hub import hf_hub_download
import os
import urllib.request
# Ensure HF transfer is enabled
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

def load_image_from_url_or_path(url_or_path: str) -> Image.Image:
    print(f"Loading image from URL or path: {url_or_path}")
    if url_or_path.startswith("http") or url_or_path.startswith("https"):
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        req = urllib.request.Request(url_or_path, headers=headers)
        return Image.open(BytesIO(urllib.request.urlopen(req).read()))
    else:
        return Image.open(url_or_path)
    
class AppInput(BaseAppInput):
    image: File
    target_width: int = 2048
    target_height: int = 2048
    prompt: str = ""
    negative_prompt: str = ""
    strength: float = 0.3
    guidance_scale: float = 7.5
    seed: int = 0

class AppOutput(BaseAppOutput):
    result: File

class App(BaseApp):
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    pipe: Optional[FluxInpaintPipeline] = None
    esrgan: Optional[RealESRGAN] = None

    async def setup(self):
        """Initialize FLUX model and RealESRGAN"""
        # Initialize FLUX
        model_id = "black-forest-labs/FLUX.1-dev"
        self.pipe = FluxInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16
        ).to(self.device)

        # Initialize RealESRGAN
        model_path = hf_hub_download(
            repo_id="ai-forever/Real-ESRGAN",
            filename="RealESRGAN_x4.pth"
        )
        self.esrgan = RealESRGAN(self.device, scale=4)
        self.esrgan.load_weights(model_path)

    async def run(self, input_data: AppInput) -> AppOutput:
        """Run upscaling with FLUX and RealESRGAN"""
        # Load input image
        image = load_image_from_url_or_path(input_data.image.path)

        def upscale_fn(img: Image.Image, scale_factor: int) -> Image.Image:
            esrgan_result = self.esrgan.predict(img)
            return esrgan_result.resize(
                (img.width * scale_factor, img.height * scale_factor),
                Image.Resampling.LANCZOS
            )

        def process_fn(img: Image.Image, mask: Image.Image) -> Image.Image:
            return self.pipe(
                prompt=input_data.prompt,
                image=img,
                mask_image=mask,
                width=img.width,
                height=img.height,
                strength=input_data.strength,
                guidance_scale=input_data.guidance_scale,
                num_inference_steps=int(10/input_data.strength),
                max_sequence_length=512,
                generator=torch.Generator(self.device).manual_seed(input_data.seed)
            ).images[0]

        # Process the image using direct upscale function
        result = upscale(
            image=image,
            target_width=input_data.target_width,
            target_height=input_data.target_height,
            tile_width=1024,
            tile_height=1024,
            redraw_padding=32,
            redraw_mask_blur=8,
            upscale_mode=UpscaleMode.CHESS,
            seam_fix_mode=SeamFixMode.NONE,
            upscale_fn=upscale_fn,
            process_fn=process_fn
        )

        # Save and return result
        output_path = "/tmp/upscaled.png"
        result.save(output_path)
        return AppOutput(result=File.from_path(output_path))

    async def unload(self):
        """Clean up resources"""
        del self.pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()