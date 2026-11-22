import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.fibo_inference.fibo2 import (
    FIBO2_PIPELINE_NAME,
    FIBO2_STYLES,
    create_fibo2_pipeline,
    default_fibo2_negative_prompt,
    load_fibo2_fonts,
    resolve_fibo2_prompt,
    run_fibo2,
)
from src.fibo_inference.inference import (
    create_pipeline,
    load_structured_prompt_input,
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
DEFAULT_RESOLUTION = "1024 1024"
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
        "--pipeline-name",
        type=str,
        default=None,
        help=f"Pipeline name to use (default: briaai/FIBO, or {FIBO2_PIPELINE_NAME} with --fibo2).",
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
        default=None,
        help=(
            f"Output resolution as 'WIDTH HEIGHT' (default: '{DEFAULT_RESOLUTION}'; "
            "with --fibo2 --edit-image, about one megapixel at the aspect ratio of the first image)."
        ),
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
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Path to the LoRA checkpoint.",
    )
    parser.add_argument(
        "--fibo-lite",
        action="store_true",
        help="Use FIBO-lite pipeline.",
    )
    fibo2 = parser.add_argument_group("Fibo 2.0", "Options for Fibo 2.0 (text-to-image and image editing).")
    fibo2.add_argument(
        "--fibo2",
        action="store_true",
        help=f"Use Fibo 2.0 ({FIBO2_PIPELINE_NAME}). Structured prompts are written by bria-vgl on Gemini.",
    )
    fibo2.add_argument(
        "--edit-image",
        action="append",
        dest="edit_images",
        default=None,
        help="Image to edit with --prompt as the instruction. Repeat for up to 4 references (<image_1>, <image_2>...).",
    )
    fibo2.add_argument(
        "--mask-path",
        default=None,
        help="Mask for a single --edit-image: white marks the region to regenerate.",
    )
    fibo2.add_argument(
        "--style",
        choices=FIBO2_STYLES,
        default=None,
        help="Photographic style for text-to-image from a short --prompt.",
    )
    fibo2.add_argument(
        "--vgl-llm",
        default=None,
        help="litellm model id bria-vgl uses to write structured prompts (default: bria-vgl's default Gemini model).",
    )
    return parser


def _load_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def main_fibo2(args: argparse.Namespace) -> None:
    """Generate or edit an image with Fibo 2.0."""
    if args.fibo_lite:
        raise SystemExit("--fibo-lite and --fibo2 cannot be combined.")
    if args.enable_teacache:
        raise SystemExit("TeaCache is not supported with --fibo2.")
    if args.model_mode == "local":
        raise SystemExit(
            "--fibo2 writes structured prompts with bria-vgl on Gemini; --model-mode local is not supported."
        )
    if args.edit_images and args.image_path:
        raise SystemExit("Use either --edit-image (edit) or --image-path (inspire), not both.")
    if args.vgl_llm is None or args.vgl_llm.startswith("gemini/"):
        if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
            raise SystemExit("bria-vgl needs GEMINI_API_KEY or GOOGLE_API_KEY set in the environment.")

    pipeline_name = args.pipeline_name or FIBO2_PIPELINE_NAME
    is_edit = bool(args.edit_images)

    width = height = None
    if args.resolution is not None or not is_edit:
        width, height = parse_resolution(args.resolution or DEFAULT_RESOLUTION)
        if width % 16 or height % 16:
            raise SystemExit("Fibo 2.0 needs a width and height divisible by 16.")

    edit_images = [_load_rgb(path) for path in args.edit_images] if is_edit else None
    mask = Image.open(args.mask_path).convert("L") if args.mask_path else None
    reference = _load_rgb(args.image_path) if args.image_path else None

    structured_prompt = args.structured_prompt
    if structured_prompt is not None:
        structured_prompt = load_structured_prompt_input(structured_prompt)
    elif args.prompt is None and reference is None and not is_edit:
        print("Generating with default prompt")
        structured_prompt = json.dumps(load_default_prompt())

    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    assert torch.cuda.is_available(), "CUDA not available"

    start_time = time.perf_counter()
    try:
        prompt_payload = resolve_fibo2_prompt(
            prompt=args.prompt,
            structured_prompt=structured_prompt,
            image=reference,
            edit_images=edit_images,
            mask=mask,
            style=args.style,
            llm=args.vgl_llm,
            seed=args.seed if args.seed >= 0 else None,
            pipeline_name=pipeline_name,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"VLM prompt generation time: {time.perf_counter() - start_time:.2f} seconds")

    caption = json.loads(prompt_payload)
    if "edit_instruction" in caption:
        print(f"edit_instruction: {caption['edit_instruction']}")
    elif "short_description" in caption:
        print(f"short_description: {caption['short_description']}")

    negative_payload = args.negative_prompt or default_fibo2_negative_prompt(
        is_edit=is_edit, style=args.style, pipeline_name=pipeline_name
    )

    pipeline = create_fibo2_pipeline(pipeline_name=pipeline_name, device="cuda", lora_path=args.lora_path)
    fonts = load_fibo2_fonts(pipeline_name)

    start_time = time.perf_counter()
    image = run_fibo2(
        pipeline=pipeline,
        prompt_payload=prompt_payload,
        negative_payload=negative_payload,
        seed=args.seed,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        images=edit_images,
        mask=mask,
        width=width,
        height=height,
        fonts=fonts,
    )
    elapsed = time.perf_counter() - start_time

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    with open(output_path.with_suffix(".json"), "w") as f:
        json.dump(caption, f, indent=2, ensure_ascii=False)
    print(f"Generation time: {elapsed:.2f} seconds")
    print(f"Saved image to {output_path}")


@torch.inference_mode()
def main():
    default_prompt = load_default_prompt()
    parser = build_parser()
    args = parser.parse_args()
    if args.num_steps <= 0:
        raise SystemExit("--steps must be a positive integer.")
    if args.fibo2:
        main_fibo2(args)
        return
    if args.edit_images or args.mask_path or args.style or args.vgl_llm:
        raise SystemExit("--edit-image, --mask-path, --style and --vgl-llm require --fibo2.")
    if args.pipeline_name is None:
        args.pipeline_name = "briaai/FIBO"
    if args.structured_prompt is None and args.prompt is None and args.image_path is None:
        print("Generating with default prompt")
        args.structured_prompt = json.dumps(default_prompt)
    if args.fibo_lite:
        args.pipeline_name = "briaai/FIBO-lite"
    if args.model_mode == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key is None:
            raise SystemExit(
                "GOOGLE_API_KEY is not set, please set it in the environment variables or switch to local mode"
            )

    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    width, height = parse_resolution(args.resolution or DEFAULT_RESOLUTION)
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

    pipeline = create_pipeline(pipeline_name=args.pipeline_name, device="cuda", lora_path=args.lora_path)

    if args.enable_teacache:
        print(f"Enabling TeaCache with threshold={args.teacache_threshold}")
        pipeline.enable_teacache(num_inference_steps=args.num_steps, rel_l1_thresh=args.teacache_threshold)

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
