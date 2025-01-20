from infsh import BaseApp, BaseAppInput, BaseAppOutput, File
from diffusers import StableDiffusionXLImg2ImgPipeline
import torch
from PIL import Image
from .upscale import Upscaler, UpscaleConfig, UpscaleMode, SeamFixMode
from typing import Optional
from io import BytesIO
from urllib.request import urlopen
from diffusers import FluxInpaintPipeline
from RealESRGAN import RealESRGAN
from huggingface_hub import hf_hub_download
import os

# Ensure HF transfer is enabled
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

def load_image_from_url_or_path(url_or_path: str) -> Image.Image:
    print(f"Loading image from URL or path: {url_or_path}")
    if url_or_path.startswith("http") or url_or_path.startswith("https"):
        return Image.open(BytesIO(urlopen(url_or_path).read()))
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

        # Create upscale configuration
        config = UpscaleConfig(
            target_width=input_data.target_width,
            target_height=input_data.target_height,
            tile_width=1024,
            tile_height=1024,
            upscale_mode=UpscaleMode.CHESS,
            seam_fix_mode=SeamFixMode.NONE,
            redraw_padding=32,
            redraw_mask_blur=8
        )

        # Create upscaler instance
        upscaler = Upscaler(config=config)

        def upscale_fn(img: Image.Image, scale_factor: int) -> Image.Image:
            esrgan_result = self.esrgan.predict(img)
            return esrgan_result.resize(
                (img.width * scale_factor, img.height * scale_factor),
                Image.Resampling.LANCZOS
            )

        def process_fn(img: Image.Image, mask: Image.Image, coords: tuple) -> Image.Image:
            # Extract region to process
            x1, y1, x2, y2 = coords
            region = img.crop(coords)
            region_mask = mask.crop(coords) if mask else None
            
            # Generate with FLUX
            output = self.pipe(
                prompt=input_data.prompt,
                image=region,
                mask_image=region_mask,
                width=region.width,
                height=region.height,
                strength=input_data.strength,
                guidance_scale=input_data.guidance_scale,
                num_inference_steps=int(10/input_data.strength),
                max_sequence_length=512,
                generator=torch.Generator(self.device).manual_seed(input_data.seed)
            ).images[0]

            # Paste back the result
            img.paste(output, (x1, y1))
            return img

        # Process the image using the Upscaler class
        result = upscaler.process_image(
            image=image,
            config=config,
            upscale_fn=upscale_fn,
            process_fn=process_fn
        )

        # Save and return result
        output_path = input_data.image.path.parent / "upscaled.png"
        result.save(output_path)
        return AppOutput(result=File(output_path))

    async def unload(self):
        """Clean up resources"""
        del self.pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()