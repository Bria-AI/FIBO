import json
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image

from src.fibo_inference.fibo_pipeline import BriaFiboPipeline
from src.fibo_inference.prompt_to_json import (
    get_json_prompt,
    load_engine,
)
from src.fibo_inference.vlm.common import DEFAULT_SAMPLING, DEFAULT_STOP_SEQUENCES, SamplingConfig

RAW_TASK_NAME = "raw"


def create_pipeline(pipeline_name: str, device: str) -> BriaFiboPipeline:
    """Initialise the FIBO pipeline on the requested device."""
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    pipe = BriaFiboPipeline.from_pretrained(
        pipeline_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    pipe.to(device=device)
    return pipe


def load_structured_prompt_input(raw_value: Optional[str]) -> str:
    """Return structured prompt text from a CLI value or JSON file path."""
    text = (raw_value or "").strip()
    if not text:
        raise SystemExit("Provide the existing structured prompt via --structured-prompt for the 'refine' task.")

    candidate_path = Path(text)
    if candidate_path.suffix.lower() == ".json":
        if not candidate_path.is_file():
            raise SystemExit(f"Structured prompt file not found: {candidate_path}")
        try:
            content = candidate_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise SystemExit(f"Failed to read structured prompt file '{candidate_path}': {exc}") from exc
        source = f"file '{candidate_path}'"
        payload = content.strip()
    else:
        source = "--structured-prompt"
        payload = text

    if not payload:
        raise SystemExit(f"The structured prompt from {source} is empty.")

    try:
        json.loads(payload)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON for {source}: {exc}") from exc

    return payload


def parse_json_string(raw_value: str, flag: str):
    """Parse a JSON string provided via CLI flags."""
    text = raw_value.strip()
    if not text:
        return ""

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON for {flag}: {exc}") from exc


def resolve_structured_prompt(
    model_mode: str,
    image_path,
    prompt,
    structured_prompt,
    stop_sequences=DEFAULT_STOP_SEQUENCES,
    temperature=DEFAULT_SAMPLING.temperature,
    top_p=DEFAULT_SAMPLING.top_p,
    max_tokens=DEFAULT_SAMPLING.max_tokens,
    device="cuda",
    vlm_model="briaai/FIBO-vlm",
) -> tuple[Any, Optional[str]]:
    """Return the structured prompt and related metadata based on CLI arguments."""
    if structured_prompt is not None and structured_prompt.endswith(".json"):
        structured_prompt = json.dumps(json.loads(open(structured_prompt).read()))
    if structured_prompt is not None and prompt is None:
        return structured_prompt
    engine = load_engine(model_mode=model_mode, model_name=vlm_model)
    sampling_config = SamplingConfig(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=stop_sequences,
    )

    if model_mode == "local":
        engine.model.to(device)

    image = Image.open(image_path) if image_path else None

    json_prompt = get_json_prompt(
        model_mode=model_mode,
        engine=engine,
        sampling_config=sampling_config,
        image=image,
        prompt=prompt,
        structured_prompt=structured_prompt,
    )

    return json_prompt


def run(
    pipeline: BriaFiboPipeline,
    prompt_payload: dict,
    negative_payload: dict,
    width: int,
    height: int,
    seed: int,
    num_steps: int,
    guidance_scale: float,
) -> Image.Image:
    assert torch.cuda.is_available()

    generator = None
    if seed >= 0:
        generator = torch.Generator(device="cuda").manual_seed(seed)

    result = pipeline(
        prompt_payload,
        num_inference_steps=num_steps,
        negative_prompt=negative_payload,
        generator=generator,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
    )

    image = result.images[0]
    return image
