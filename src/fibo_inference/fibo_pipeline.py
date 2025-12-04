# Copyright (c) Bria.ai. All rights reserved.
#
# This file is licensed under the Creative Commons Attribution-NonCommercial 4.0 International Public License (CC-BY-NC-4.0).
# You may obtain a copy of the license at https://creativecommons.org/licenses/by-nc/4.0/
#
# You are free to share and adapt this material for non-commercial purposes provided you give appropriate credit,
# indicate if changes were made, and do not use the material for commercial purposes.
#
# See the license for further details.


from diffusers import BriaFiboPipeline as DiffusersBriaFiboPipeline
from diffusers.utils import (
    logging,
)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Example:
    ```python
    import torch
    from diffusers import BriaFiboPipeline
    from diffusers.modular_pipelines import ModularPipeline

    torch.set_grad_enabled(False)
    vlm_pipe = ModularPipeline.from_pretrained("briaai/FIBO-VLM-prompt-to-JSON", trust_remote_code=True)

    pipe = BriaFiboPipeline.from_pretrained(
        "briaai/FIBO",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()

    with torch.inference_mode():
        # 1. Create a prompt to generate an initial image
        output = vlm_pipe(prompt="a beautiful dog")
        json_prompt_generate = output.values["json_prompt"]

        # Generate the image from the structured json prompt
        results_generate = pipe(prompt=json_prompt_generate, num_inference_steps=50, guidance_scale=5)
        results_generate.images[0].save("image_generate.png")
    ```
"""


class BriaFiboPipeline(DiffusersBriaFiboPipeline):
    def enable_teacache(self, num_inference_steps: int, rel_l1_thresh: float = 1.0):
        """
        Enable TeaCache for faster inference by caching and reusing intermediate computations.

        TeaCache monitors the change in hidden states between denoising steps and reuses
        cached computations when changes are below a threshold, significantly speeding up
        inference with minimal quality loss.

        Args:
            num_inference_steps (int): Total number of denoising steps that will be used.
            rel_l1_thresh (float, optional): Threshold for cache reuse decision.
                Higher values result in more aggressive caching (faster, potentially lower quality).
                Lower values result in more conservative caching (slower, better quality).
                Defaults to 1.0. Recommended range: 0.6-1.0.

        Example:
            >>> pipe.enable_teacache(num_inference_steps=50, rel_l1_thresh=1.0)
            >>> result = pipe(prompt="...", num_inference_steps=50)

        Note:
            Make sure the num_inference_steps matches what you'll use in the __call__ method.
        """
        from .teacache import teacache_forward

        # Store the original forward method if not already stored
        if not hasattr(self.transformer.__class__, "_original_forward"):
            self.transformer.__class__._original_forward = self.transformer.__class__.forward

        # Replace forward with teacache version
        self.transformer.__class__.forward = teacache_forward

        # Initialize TeaCache state variables
        self.transformer.__class__.enable_teacache = True
        self.transformer.__class__.num_steps = num_inference_steps
        self.transformer.__class__.rel_l1_thresh = rel_l1_thresh
        self.transformer.__class__.cnt = 0
        self.transformer.__class__.accumulated_rel_l1_distance = 0.0
        self.transformer.__class__.previous_modulated_input = None
        self.transformer.__class__.previous_residual = None

        logger.info(f"TeaCache enabled: num_steps={num_inference_steps}, threshold={rel_l1_thresh}")

    def disable_teacache(self):
        """
        Disable TeaCache and restore the original forward method.

        This cleans up TeaCache state and restores normal inference behavior.

        Example:
            >>> pipe.disable_teacache()
        """
        # Restore original forward method
        if hasattr(self.transformer.__class__, "_original_forward"):
            self.transformer.__class__.forward = self.transformer.__class__._original_forward

        # Clean up TeaCache state
        self.transformer.__class__.enable_teacache = False
        self.transformer.__class__.cnt = 0
        self.transformer.__class__.accumulated_rel_l1_distance = 0.0
        self.transformer.__class__.previous_modulated_input = None
        self.transformer.__class__.previous_residual = None

        logger.info("TeaCache disabled")
