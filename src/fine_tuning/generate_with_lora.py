import json
import os
import ujson

import torch
from diffusers import BriaFiboPipeline
from src.fine_tuning.lora_utils import set_lora_training, load_lora

prompt = "A blue bear wears a black tuxedo with a black bow tie. It stands upright beside a tall bar counter and holds a clear martini glass in one paw"
prompt = {'short_description': f'{prompt}'}

prompt_short = ujson.dumps(prompt[0], escape_forward_slashes=False)

# Load the FIBO pipeline
pipe = BriaFiboPipeline.from_pretrained(
    "briaai/FIBO",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

set_lora_training(None,transformer=pipe.transformer, lora_rank=128)
load_lora(transformer=pipe.transformer, input_dir="/home/ubuntu/enhanced_bears/results_adamw_fixed/checkpoint_400/")

height = 1024
width = 1024

results = pipe(
    prompt=prompt_short, num_inference_steps=50, guidance_scale=5,
    height=height, width=width,
)
img = results.images[0]
