from infsh import BaseApp, BaseAppInput, BaseAppOutput, File
from diffusers import StableDiffusionXLImg2ImgPipeline
import torch
from PIL import Image
from .upscale import Upscaler, UpscaleConfig, UpscaleMode, SeamFixMode
from typing import Optional
from io import BytesIO
from urllib.request import urlopen

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

class AppOutput(BaseAppOutput):
    result: File

class App(BaseApp):
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    pipe: Optional[StableDiffusionXLImg2ImgPipeline] = None

    async def setup(self):
        """Initialize SDXL model"""
        self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)

    async def run(self, input_data: AppInput) -> AppOutput:
        """Run upscaling with SDXL"""
        # Load input image
        image = load_image_from_url_or_path(input_data.image.path)

        # Create upscale configuration
        config = UpscaleConfig(
            target_width=input_data.target_width,
            target_height=input_data.target_height,
            tile_width=1024,  # SDXL's optimal tile size
            tile_height=1024,
            upscale_mode=UpscaleMode.CHESS,
            seam_fix_mode=SeamFixMode.HALF_TILE_PLUS_INTERSECTIONS
        )

        # Create upscaler instance
        upscaler = Upscaler(config=config)

        def upscale_fn(img: Image.Image, scale_factor: int) -> Image.Image:
            # Simple resize for initial upscaling
            return img.resize(
                (img.width * scale_factor, img.height * scale_factor),
                Image.LANCZOS
            )

        def process_fn(img: Image.Image, mask: Image.Image, coords: tuple) -> Image.Image:
            # Extract region to process
            x1, y1, x2, y2 = coords
            region = img.crop(coords)
            
            # Generate with SDXL
            output = self.pipe(
                prompt=input_data.prompt,
                negative_prompt=input_data.negative_prompt,
                image=region,
                strength=input_data.strength,
                guidance_scale=input_data.guidance_scale,
                num_inference_steps=30,
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