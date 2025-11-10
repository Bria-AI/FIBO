import argparse
import json
import os

from diffusers import BriaFiboPipeline
from diffusers.loaders import FluxLoraLoaderMixin
import torch
import ujson

RESOLUTIONS_WH = [
    "832 1248",
    "896 1152",
    "960 1088",
    "1024 1024",
    "1088 960",
    "1152 896",
    "1216 832",
    "1280 800",
    "1344 768",
]


class BriaFiboPipelineWithLoRA(FluxLoraLoaderMixin, BriaFiboPipeline):
    r"""
    A BriaFiboPipeline with LoRA (Low-Rank Adaptation) support.
    
    This class extends BriaFiboPipeline with LoRA loading capabilities, allowing you to:
    - Load LoRA weights using `load_lora_weights()`
    - Save LoRA weights using `save_lora_weights()`
    - Unload LoRA weights using `unload_lora_weights()`
    - Enable/disable LoRA adapters
    - Fuse/unfuse LoRA weights
    
    All other functionality from BriaFiboPipeline remains unchanged.
    
    Example:
        ```python
        from FIBO.src.fibo_inference.fibo_pipeline import BriaFiboPipelineWithLoRA
        
        pipe = BriaFiboPipelineWithLoRA.from_pretrained(
            "briaai/FIBO",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        pipe.load_lora_weights("path/to/lora/weights")
        ```
    """


def parse_resolution(raw_value: str) -> tuple[int, int]:
    """Parse resolution in the form 'WIDTH HEIGHT'."""
    normalised = raw_value.replace(",", " ").replace("x", " ")
    parts = [part for part in normalised.split() if part]
    if len(parts) != 2:
        raise SystemExit("Resolution must contain exactly two integers, e.g. '1024 1024'.")

    try:
        width, height = (int(parts[0]), int(parts[1]))
    except ValueError as exc:
        raise SystemExit("Resolution values must be integers.") from exc

    if width <= 0 or height <= 0:
        raise SystemExit("Resolution values must be positive.")

    return width, height


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images with LoRA weights")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="briaai/FIBO",
        help="Path to pretrained model or model identifier from Hugging Face"
    )
    parser.add_argument(
        "--lora_ckpt_path",
        type=str,
        required=True,
        help="Path to LoRA checkpoint directory"
    )
    parser.add_argument(
        "--structered_prompt_path",
        type=str,
        required=True,
        help="Path to structured prompt JSON file"
    )
    parser.add_argument(
        "--output_image_path",
        type=str,
        default="generated_image.png",
        help="Path to save the generated image (default: generated_image.png)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the random number generator (default: 42)"
    )
    parser.add_argument(
        "--resolution",
        default="1024 1024",
        help="Output resolution as 'WIDTH HEIGHT'.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # load json prompt
    with open(args.structered_prompt_path, "r") as f:
        prompt = json.load(f)
    prompt = ujson.dumps(prompt, escape_forward_slashes=False)

    # Load the FIBO pipeline
    pipe = BriaFiboPipelineWithLoRA.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")
    pipe.load_lora_weights(args.lora_ckpt_path)

    width, height = parse_resolution(args.resolution)
    if f"{width} {height}" not in RESOLUTIONS_WH:
        print(f"Note: {width}x{height} is outside the preset resolutions used by the original demo.")
    
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    results = pipe(
        prompt=prompt, num_inference_steps=50, guidance_scale=5,
        height=height, width=width, generator=generator
    )
    img = results.images[0]
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_image_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Save the image
    img.save(args.output_image_path)
    print(f"Image saved to {args.output_image_path}")
    
    return img


if __name__ == "__main__":
    main()
