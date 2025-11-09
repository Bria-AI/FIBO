import argparse
import json

from diffusers import BriaFiboPipeline
from diffusers.loaders import FluxLoraLoaderMixin
import torch
import ujson


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

    transformer_name = "transformer"
    text_encoder_name = "text_encoder"


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

    height = 1024
    width = 1024

    results = pipe(
        prompt=prompt, num_inference_steps=50, guidance_scale=5,
        height=height, width=width,
    )
    img = results.images[0]
    
    return img


if __name__ == "__main__":
    main()
