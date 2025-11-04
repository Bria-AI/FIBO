import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from src.fibo_inference.inference import (
    create_pipeline,
    resolve_structured_prompt,
    run,
)
from src.fibo_inference.parse_caption import clean_json, prepare_clean_caption
from src.fibo_inference.vlm.common import DEFAULT_SAMPLING, DEFAULT_STOP_SEQUENCES

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

DEFAULT_STEPS = 50
DEFAULT_OUTPUT_PATH = "output/generated.png"


def get_default_negative_prompt(existing_json: dict) -> str:
    negative_prompt = ""
    style_medium = existing_json.get("style_medium", "").lower()
    if style_medium in ["photograph", "photography", "photo"]:
        negative_prompt = """{'style_medium':'digital illustration','artistic_style':'non-realistic'}"""
    return negative_prompt


def load_default_prompt() -> dict:
    """Load and normalise the default caption JSON used by the Gradio demo."""
    default_path = Path("default_json_caption.json")
    with default_path.open() as f:
        data = json.load(f)

    data["pickascore"] = 1.0
    data["aesthetic_score"] = 10.0
    cleaned = prepare_clean_caption(data)
    return json.loads(cleaned)


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


def build_parser() -> argparse.ArgumentParser:
    """Configure the CLI parser."""
    parser = argparse.ArgumentParser(description="Generate images with the FIBO model.")
    parser.add_argument(
        "--pipeline-name", type=str, default="briaai/FIBO", help="Pipeline name to use."
    )
    parser.add_argument("--vlm-model", type=str, default="briaai/FIBO-vlm", help="VLM model to use.")
    parser.add_argument("--model-mode", choices=["local", "gemini"], default="gemini", help="Model mode to use.")
    parser.add_argument(
        "--negative-prompt",
        default="",
        help="Negative prompt",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Random seed. Use -1 for a random seed on each run.",
    )
    parser.add_argument(
        "--resolution",
        default="1024 1024",
        help="Output resolution as 'WIDTH HEIGHT'.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=5.0,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        dest="num_steps",
        help="Number of inference steps.",
    )
    parser.add_argument(
        "--prompt",
        help="Short natural-language prompt.",
    )
    parser.add_argument(
        "--structured-prompt",
        help="Existing structured prompt.",
    )
    parser.add_argument(
        "--image-path",
        help="Path to the image to be used by the VLM.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_SAMPLING.temperature,
        help="Override temperature for VLM prompt generation (optional).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=DEFAULT_SAMPLING.top_p,
        help="Override top-p for VLM prompt generation (optional).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_SAMPLING.max_tokens,
        help="Override max tokens for VLM prompt generation (optional).",
    )
    parser.add_argument(
        "--stop-sequence",
        action="append",
        dest="stop_sequences",
        default=DEFAULT_STOP_SEQUENCES,
        help=("Custom stop sequence for VLM prompt generation (repeat for multiple values)."),
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output image path (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--enable-teacache",
        action="store_true",
        help="Enable TeaCache for faster inference with minimal quality loss.",
    )
    parser.add_argument(
        "--teacache-threshold",
        type=float,
        default=1.0,
        help="TeaCache threshold (0.6-1.0). Higher = faster but potentially lower quality. Default: 1.0",
    )
    return parser


@torch.inference_mode()
def main():
    default_prompt = load_default_prompt()
    parser = build_parser()
    args = parser.parse_args()
    if args.structured_prompt is None and args.prompt is None and args.image_path is None:
        print("Generating with default prompt")
        args.structured_prompt = json.dumps(default_prompt)

    if args.model_mode == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key is None:
            raise SystemExit(
                "GOOGLE_API_KEY is not set, please set it in the environment variables or switch to local mode"
            )
    if args.num_steps <= 0:
        raise SystemExit("--steps must be a positive integer.")

    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    width, height = parse_resolution(args.resolution)
    if f"{width} {height}" not in RESOLUTIONS_WH:
        print(f"Note: {width}x{height} is outside the preset resolutions used by the original demo.")

    assert torch.cuda.is_available(), "CUDA not available"

    start_time = time.perf_counter()
    if args.structured_prompt is not None and args.prompt is None and args.image_path is None:
        # json input and no other input -- skip VLM
        if args.structured_prompt.endswith(".json"):
            json_prompt = json.loads(open(args.structured_prompt).read())
        else:
            json_prompt = json.loads(args.structured_prompt)
    else:
        json_prompt = resolve_structured_prompt(
            model_mode=args.model_mode,
            device="cuda",
            vlm_model=args.vlm_model,
            image_path=args.image_path,
            prompt=args.prompt,
            structured_prompt=args.structured_prompt,
            stop_sequences=args.stop_sequences,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
    elapsed = time.perf_counter() - start_time
    print(f"VLM prompt generation time: {elapsed:.2f} seconds")

    prompt_payload = clean_json(json_prompt)
    negative_payload = args.negative_prompt
    if negative_payload == "":
        negative_payload = get_default_negative_prompt(json.loads(prompt_payload))

    pipeline = create_pipeline(pipeline_name=args.pipeline_name, device="cuda")
    
    if args.enable_teacache:
        print(f"Enabling TeaCache with threshold={args.teacache_threshold}")
        pipeline.enable_teacache(
            num_inference_steps=args.num_steps,
            rel_l1_thresh=args.teacache_threshold
        )
    
    if isinstance(json_prompt, dict) and "short_description" in json_prompt:
        print(f"short_description: {json_prompt['short_description']}")

    start_time = time.perf_counter()
    image = run(
        pipeline=pipeline,
        prompt_payload=prompt_payload,
        negative_payload=negative_payload,
        width=width,
        height=height,
        seed=args.seed,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
    )
    elapsed = time.perf_counter() - start_time

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    # dump json_prompt to a file
    with open(output_path.with_suffix(".json"), "w") as f:
        json.dump(json.loads(prompt_payload), f, indent=2)
    print(f"Generation time: {elapsed:.2f} seconds")
    print(f"Saved image to {output_path}")


if __name__ == "__main__":
    main()
