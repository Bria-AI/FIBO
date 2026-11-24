"""Fibo 2.0 support: one model for text-to-image and image editing.

Fibo 2.0 (https://huggingface.co/briaai/Fibo-2.0-dev) reads structured VGL captions. The captions are written by
`bria-vgl` (https://github.com/Bria-AI/vgl), which runs on Gemini, and rendered by diffusers' `BriaFibo2Pipeline`.
"""

import json
from typing import Optional, Sequence, Union

import torch
from diffusers import BriaFibo2Pipeline
from huggingface_hub import hf_hub_download
from PIL import Image

FIBO2_PIPELINE_NAME = "briaai/Fibo-2.0-dev"
FIBO2_FONTS = ("Aileron-Regular.otf", "DejaVuSans.ttf")
FIBO2_STYLES = ("boring_reality", "kodak_film", "hard_flash", "hazy_dreamy", "lens_distortion")
# The LLM the styles were tuned with, per the model card.
FIBO2_STYLE_LLM = "gemini/gemini-3.5-flash"
FIBO2_STYLE_MAX_OUTPUT_TOKENS = 16000
FIBO2_MAX_EDIT_IMAGES = 4

# Negative prompts from the model card. Text-to-image steers away from soft, featureless, artifact-laden images;
# edits use the empty caption the model was trained with.
NEGATIVE_T2I = json.dumps(
    {
        "short_description": "",
        "objects": [
            {
                "description": "the main subject",
                "texture": "smooth, featureless, plastic-like, airbrushed, no fine surface detail",
                "appearance_details": "soft blurred edges, simplified shapes, no fine detail",
                "skin_tone_and_texture": "waxy airbrushed skin, poreless, plastic smooth",
            }
        ],
        "background_setting": (
            "a plain seamless studio backdrop, perfectly smooth gradient, blurred and featureless, no visible texture"
        ),
        "lighting": {},
        "aesthetics": {
            "artifacts": [
                "oversaturated colours",
                "neon colour cast",
                "overexposed highlights",
                "lens flare",
                "slight blur",
                "motion blur",
                "haze",
            ]
        },
        "photographic_characteristics": {
            "depth_of_field": "shallow, heavy bokeh obscuring most of the frame",
            "focus": "soft, slightly out of focus, lacking fine texture",
        },
        "style_medium": "",
        "context": "",
        "artistic_style": "",
    },
    separators=(",", ":"),
)
NEGATIVE_EDIT = json.dumps(
    {
        "short_description": "",
        "objects": [],
        "background_setting": "",
        "lighting": {},
        "aesthetics": {},
        "photographic_characteristics": {},
        "style_medium": "",
        "context": "",
        "artistic_style": "",
    },
    separators=(",", ":"),
)


def create_fibo2_pipeline(
    pipeline_name: str = FIBO2_PIPELINE_NAME, device: str = "cuda", lora_path: Optional[str] = None
) -> BriaFibo2Pipeline:
    """Load the Fibo 2.0 pipeline on the requested device."""
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    pipe = BriaFibo2Pipeline.from_pretrained(pipeline_name, torch_dtype=dtype)
    pipe.to(device=device)
    if lora_path:
        pipe.load_lora_weights(lora_path)
    return pipe


def load_fibo2_fonts(pipeline_name: str = FIBO2_PIPELINE_NAME) -> list[str]:
    """Download the fonts Fibo 2.0 was trained with, used to draw the text in a caption."""
    return [hf_hub_download(pipeline_name, name, subfolder="fonts") for name in FIBO2_FONTS]


def load_fibo2_style_file(style: str, name: str, pipeline_name: str = FIBO2_PIPELINE_NAME) -> str:
    """Read one of a style's files (`system_prompt.txt`, `user_prompt.txt` or `negative_prompt.json`)."""
    if style not in FIBO2_STYLES:
        raise ValueError(f"Unknown Fibo 2.0 style '{style}'. Choose from: {', '.join(FIBO2_STYLES)}")
    path = hf_hub_download(pipeline_name, name, subfolder=f"styles/{style}")
    with open(path, encoding="utf-8") as f:
        return f.read()


def default_fibo2_negative_prompt(is_edit: bool, style: Optional[str] = None, pipeline_name: str = FIBO2_PIPELINE_NAME):
    """The negative prompt the model card recommends for the mode."""
    if is_edit:
        return NEGATIVE_EDIT
    if style:
        return load_fibo2_style_file(style, "negative_prompt.json", pipeline_name)
    return NEGATIVE_T2I


def _vgl_to_json(scene) -> str:
    return json.dumps(scene.model_dump(exclude_none=True), ensure_ascii=False)


def _load_input_vgl(structured_prompt: Union[str, dict]):
    from bria_vgl.dialects.thales import ThalesVGL

    data = json.loads(structured_prompt) if isinstance(structured_prompt, str) else structured_prompt
    return ThalesVGL.model_validate(data)


def resolve_fibo2_prompt(
    prompt: Optional[str] = None,
    structured_prompt: Optional[str] = None,
    image: Optional[Image.Image] = None,
    edit_images: Optional[Sequence[Image.Image]] = None,
    mask: Optional[Image.Image] = None,
    style: Optional[str] = None,
    llm: Optional[str] = None,
    seed: Optional[int] = None,
    pipeline_name: str = FIBO2_PIPELINE_NAME,
) -> str:
    """Write the structured VGL caption Fibo 2.0 renders, with `bria-vgl`.

    - `edit_images` (+ optional `mask`) and `prompt`: an edit instruction for the images. A `structured_prompt` is used
      as the VGL of the first image, so the edit keeps what the prompt does not change.
    - `image` with a `prompt`: Inspire, a new scene borrowing from the reference image. Without a prompt: a caption of
      the image, to regenerate a variation of it.
    - `structured_prompt` and `prompt`: Refine the structured prompt with the instruction.
    - `structured_prompt` alone: used as is.
    - `prompt` alone: Generate, optionally in one of the photographic `style`s.

    Returns the caption as a JSON string.
    """
    import bria_vgl
    from bria_vgl.dialects.thales import Caption, EditImages, EditImageWithMask, Generate, Inspire, Refine

    llm_kwargs = {"seed": seed}
    if llm:
        llm_kwargs["llm"] = llm

    if style and (edit_images or image is not None or structured_prompt is not None or not prompt):
        raise ValueError("Fibo 2.0 styles apply to text-to-image from a short prompt only (--prompt alone).")

    if edit_images:
        if not prompt:
            raise ValueError("Editing needs an instruction (--prompt).")
        input_image_vgl = _load_input_vgl(structured_prompt) if structured_prompt is not None else None
        if mask is not None:
            if len(edit_images) != 1:
                raise ValueError("A mask can only be used when editing a single image.")
            payload = EditImageWithMask(prompt=prompt, image=edit_images[0], mask=mask, input_image_vgl=input_image_vgl)
        else:
            if len(edit_images) > FIBO2_MAX_EDIT_IMAGES:
                raise ValueError(f"Fibo 2.0 edits take up to {FIBO2_MAX_EDIT_IMAGES} images.")
            payload = EditImages(prompt=prompt, images=list(edit_images), input_image_vgl=input_image_vgl)
        return _vgl_to_json(bria_vgl.modify(payload, **llm_kwargs))

    if mask is not None:
        raise ValueError("A mask needs an image to edit (--edit-image).")

    if image is not None:
        payload = Inspire(prompt=prompt, image=image) if prompt else Caption(image=image)
        return _vgl_to_json(bria_vgl.create(payload, **llm_kwargs))

    if structured_prompt is not None:
        if not prompt:
            return json.dumps(json.loads(structured_prompt), ensure_ascii=False)
        payload = Refine(prompt=prompt, input_vgl=_load_input_vgl(structured_prompt))
        return _vgl_to_json(bria_vgl.modify(payload, **llm_kwargs))

    if not prompt:
        raise ValueError("Provide a prompt, a structured prompt, an image or images to edit.")

    create_kwargs = {}
    if style:
        create_kwargs = {
            "llm": llm or FIBO2_STYLE_LLM,
            "system_prompt": load_fibo2_style_file(style, "system_prompt.txt", pipeline_name),
            "final_prompt": load_fibo2_style_file(style, "user_prompt.txt", pipeline_name),
            "max_output_tokens": FIBO2_STYLE_MAX_OUTPUT_TOKENS,
        }
    return _vgl_to_json(bria_vgl.create(Generate(prompt=prompt), **{**llm_kwargs, **create_kwargs}))


def run_fibo2(
    pipeline: BriaFibo2Pipeline,
    prompt_payload: str,
    negative_payload: str,
    seed: int,
    num_steps: int,
    guidance_scale: float,
    images: Optional[Sequence[Image.Image]] = None,
    mask: Optional[Image.Image] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    fonts: Optional[list[str]] = None,
) -> Image.Image:
    """Render a caption with Fibo 2.0, or apply an edit to `images` when given."""
    generator = None
    if seed >= 0:
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)

    image_arg = None
    if images:
        image_arg = images[0] if len(images) == 1 else list(images)

    result = pipeline(
        prompt=prompt_payload,
        negative_prompt=negative_payload,
        image=image_arg,
        mask=mask,
        width=width,
        height=height,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        fonts=fonts,
    )
    return result.images[0]
