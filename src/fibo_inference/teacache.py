# teacache.py
from typing import Any, Dict, Optional, Union
import numpy as np
import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_version,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from .transformer_fibo import BriaFiboTransformer2DModel

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@torch.no_grad()
def _checkpoint_or_run_block(module, *inputs):
    if torch.is_grad_enabled():
        def create_custom_forward(mod):
            def custom_forward(*x):
                return mod(*x)
            return custom_forward
        ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
        return torch.utils.checkpoint.checkpoint(create_custom_forward(module), *inputs, **ckpt_kwargs)
    else:
        return module(*inputs)


def teacache_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    text_encoder_layers: Optional[list] = None,
    pooled_projections: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    guidance: torch.Tensor = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
    """
    TeaCache-enabled forward for BriaFiboTransformer2DModel.
    """

    # ----- LoRA scaling parity -----
    if joint_attention_kwargs is not None:
        joint_attention_kwargs = dict(joint_attention_kwargs)
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)
    else:
        if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
            logger.warning("Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective.")

    # ----- Input embeddings -----
    # Token/latent embedding
    hidden_states = self.x_embedder(hidden_states)

    timestep = timestep.to(hidden_states.dtype)
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype)
    else:
        guidance = None

    temb = self.time_embed(timestep, dtype=hidden_states.dtype)

    # Optional guidance embedding
    if guidance is not None:
        temb = temb + self.guidance_embed(guidance, dtype=hidden_states.dtype)

    # Context/text embedding path
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    # Build rotary embeddings from ids
    if len(txt_ids.shape) == 3:
        txt_ids = txt_ids[0]
    if len(img_ids.shape) == 3:
        img_ids = img_ids[0]
    ids = torch.cat((txt_ids, img_ids), dim=0)
    image_rotary_emb = self.pos_embed(ids)

    # Project text_encoder_layers through caption_projection
    new_text_encoder_layers = []
    for i, text_encoder_layer in enumerate(text_encoder_layers):
        text_encoder_layer = self.caption_projection[i](text_encoder_layer)
        new_text_encoder_layers.append(text_encoder_layer)
    text_encoder_layers = new_text_encoder_layers

    # ===== TeaCache indicator (first block norm1 modulation) =====
    should_calc = True
    if getattr(self.__class__, "enable_teacache", False):
        inp = hidden_states.clone()
        temb_ = temb.clone()
        # Blocks follow DiT-like norm1(modulate) -> attention/mlp; probe the first block's norm
        modulated_inp, *_ = self.transformer_blocks[0].norm1(inp, emb=temb_)

        if self.__class__.cnt == 0 or self.__class__.cnt == self.__class__.num_steps - 1:
            should_calc = True
            self.__class__.accumulated_rel_l1_distance = 0.0
        else:
            prev = self.__class__.previous_modulated_input
            denom = prev.abs().mean().clamp_min(1e-12)
            rel_l1 = (modulated_inp - prev).abs().mean() / denom

            # Polynomial rescale (same default for Flux)
            coefficients = [4.98651651e02, -2.83781631e02, 5.58554382e01, -3.82021401e00, 2.64230861e-01]
            rescale = float(np.poly1d(coefficients)(float(rel_l1)))

            self.__class__.accumulated_rel_l1_distance += rescale
            if self.__class__.accumulated_rel_l1_distance < self.__class__.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.__class__.accumulated_rel_l1_distance = 0.0

        self.__class__.previous_modulated_input = modulated_inp
        self.__class__.cnt += 1
        if self.__class__.cnt == self.__class__.num_steps:
            self.__class__.cnt = 0

    # ===== Cached reuse or full compute =====
    if getattr(self.__class__, "enable_teacache", False) and not should_calc:
        if getattr(self.__class__, "previous_residual", None) is None:
            should_calc = True  # first usable step still needs a compute
        else:
            hidden_states = hidden_states + self.__class__.previous_residual

    if should_calc:
        # -------- multi-block tower --------
        block_id = 0
        for index_block, block in enumerate(self.transformer_blocks):
            # FIBO: concatenate text_encoder_layer into encoder_hidden_states before block call
            current_text_encoder_layer = text_encoder_layers[block_id]
            encoder_hidden_states = torch.cat([
                encoder_hidden_states[:, :, :self.inner_dim // 2], current_text_encoder_layer
            ], dim=-1)
            block_id += 1
            
            encoder_hidden_states, hidden_states = _checkpoint_or_run_block(
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                image_rotary_emb,
                joint_attention_kwargs,
            )

        # -------- single-block tower --------
        for index_block, block in enumerate(self.single_transformer_blocks):
            # FIBO: concatenate text_encoder_layer, then concat encoder+hidden states
            current_text_encoder_layer = text_encoder_layers[block_id]
            encoder_hidden_states = torch.cat([
                encoder_hidden_states[:, :, :self.inner_dim // 2], current_text_encoder_layer
            ], dim=-1)
            block_id += 1
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            
            hidden_states = _checkpoint_or_run_block(
                block,
                hidden_states,
                temb,
                image_rotary_emb,
                joint_attention_kwargs,
            )
            
            encoder_hidden_states = hidden_states[:, : encoder_hidden_states.shape[1], ...]
            hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

        # Cache residual wrt. the pre-tower input
        if getattr(self.__class__, "enable_teacache", False):
            # 'inp' was defined above only inside TeaCache enable branch
            base_inp = inp if "inp" in locals() else hidden_states.new_zeros_like(hidden_states)
            self.__class__.previous_residual = hidden_states - base_inp

    # ----- output projection -----
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)
