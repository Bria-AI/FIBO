import json
from typing import Optional, Union

from google import genai
from PIL import Image

from src.fibo_inference.vlm.common import DEFAULT_SAMPLING, SamplingConfig
from src.fibo_inference.vlm.gemini_api import json_promptify
from src.fibo_inference.vlm.local_vlm import TransformersEngine, build_messages


def load_engine(
    model_mode: str = "local",
    model_name: str = "briaai/vlm-processor",
) -> Union[TransformersEngine, genai.Client]:
    """Load the VLM model"""
    if model_mode == "local":
        engine = TransformersEngine(model=model_name)
    elif model_mode == "gemini":
        engine = genai.Client()
    else:
        raise ValueError(f"Invalid model mode: {model_mode}")

    return engine


def run_local_engine(
    engine: TransformersEngine,
    image,
    prompt,
    structured_prompt,
    sampling_config: SamplingConfig,
):
    refine_image = None
    if image is None and structured_prompt is None:
        # only got prompt
        task = "generate"
        editing_instructions = None
    elif image is None and structured_prompt is not None and prompt is not None:
        # got structured prompt and prompt
        task = "refine"
        editing_instructions = prompt
    elif image is not None and structured_prompt is None and prompt is not None:
        # got image and prompt
        task = "refine"
        editing_instructions = prompt
        refine_image = image
    elif image is not None and structured_prompt is None and prompt is None:
        # only got image
        task = "inspire"
        editing_instructions = None
    else:
        raise ValueError("Invalid input")

    messages = build_messages(
        task,
        image=image,
        refine_image=refine_image,
        prompt=prompt,
        structured_prompt=structured_prompt,
        editing_instructions=editing_instructions,
    )

    generated_prompt = engine.generate(
        messages=messages,
        top_p=sampling_config.top_p,
        temperature=sampling_config.temperature,
        max_tokens=sampling_config.max_tokens,
        stop=sampling_config.stop,
    )
    return generated_prompt


def get_json_prompt(
    engine: Union[TransformersEngine, genai.Client],
    model_mode: str,
    *,
    image: Optional[Image.Image] = None,
    prompt: Optional[str] = None,
    structured_prompt: Optional[str] = None,
    sampling_config: Optional[SamplingConfig] = None,
    model_id: str = "gemini-2.5-flash",
) -> str:
    """Run the requested task and return the generated structured prompt."""

    if model_mode == "local":
        return run_local_engine(
            engine=engine,
            image=image,
            prompt=prompt,
            structured_prompt=structured_prompt,
            sampling_config=sampling_config or DEFAULT_SAMPLING,
        )
    elif model_mode == "gemini":
        json_data = json_promptify(
            client=engine,
            user_prompt=prompt,
            image=image,
            existing_json=structured_prompt,
            top_p=sampling_config.top_p,
            temperature=sampling_config.temperature,
            max_tokens=sampling_config.max_tokens,
            model_id=model_id,
        )
        json_data = json.loads(json_data)
    return json_data
