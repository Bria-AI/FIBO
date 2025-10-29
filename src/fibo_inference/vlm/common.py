from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float
    top_p: float
    max_tokens: int
    stop: Optional[List[str]] = None


DEFAULT_STOP_SEQUENCES: List[str] = ["<|im_end|>", "<|end_of_text|>"]

DEFAULT_SAMPLING = SamplingConfig(
    temperature=0.2,
    top_p=0.9,
    max_tokens=4096,
    stop=DEFAULT_STOP_SEQUENCES,
)
