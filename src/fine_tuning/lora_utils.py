import torch

from diffusers.loaders.lora_pipeline import FluxLoraLoaderMixin
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_unet_state_dict_to_peft
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict


def add_lora(transformer, lora_rank):
    target_modules = [
            # HF Lora Layers
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
            "proj_mlp",
            # +  layers that exist on ostris ai-toolkit / replicate trainer
            "norm1_context.linear", "norm1.linear","norm.linear","proj_out"
        ]
    transformer_lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer.add_adapter(transformer_lora_config)

def set_lora_training(accelerator, transformer,lora_rank):
    add_lora(transformer, lora_rank)
    
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None

            for model in models:
                if isinstance(model, type(accelerator.unwrap_model(transformer))):
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            FluxLoraLoaderMixin.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
            )
    

    def load_model_hook(models, input_dir):
        transformer_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(accelerator.unwrap_model(transformer))):
                transformer_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        load_lora(transformer=transformer_,input_dir=input_dir)
        # Make sure the trainable params are in float32. This is again needed since the base models
        cast_training_params([transformer_], dtype=torch.float32)
    if accelerator:
        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format

def load_lora(transformer,input_dir):
    lora_state_dict = FluxLoraLoaderMixin.lora_state_dict(input_dir)

    transformer_state_dict = {
        f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("transformer.")
    }
    transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
    incompatible_keys = set_peft_model_state_dict(transformer, transformer_state_dict, adapter_name="default")
    if incompatible_keys is not None:
        # check only for unexpected keys
        unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
        if unexpected_keys:
            raise Exception(f"Loading adapter weights from state_dict led to unexpected keys not found in the model: {unexpected_keys}. "
)