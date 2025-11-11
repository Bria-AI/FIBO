import abc
import os
from typing import List, Union

import torch
from diffusers.loaders.lora_pipeline import FluxLoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_unet_state_dict_to_peft, logging
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


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
        "norm1_context.linear",
        "norm1.linear",
        "norm.linear",
        "proj_out",
    ]
    transformer_lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer.add_adapter(transformer_lora_config)


def set_lora_training(accelerator, transformer, lora_rank):
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

        load_lora(transformer=transformer_, input_dir=input_dir)
        # Make sure the trainable params are in float32. This is again needed since the base models
        cast_training_params([transformer_], dtype=torch.float32)

    if accelerator:
        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format


def load_lora(transformer, input_dir):
    lora_state_dict = FluxLoraLoaderMixin.lora_state_dict(input_dir)

    transformer_state_dict = {
        f"{k.replace('transformer.', '')}": v for k, v in lora_state_dict.items() if k.startswith("transformer.")
    }
    transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
    incompatible_keys = set_peft_model_state_dict(transformer, transformer_state_dict, adapter_name="default")
    if incompatible_keys is not None:
        # check only for unexpected keys
        unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
        if unexpected_keys:
            raise Exception(
                f"Loading adapter weights from state_dict led to unexpected keys not found in the model: {unexpected_keys}. "
            )


# Not really cosine but with decay
def get_cosine_schedule_with_warmup_and_decay(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
    constant_steps=-1,
    eps=1e-5,
) -> LambdaLR:
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_periods (`float`, *optional*, defaults to 0.5):
            The number of periods of the cosine function in a schedule (the default is to just decrease from the max
            value to 0 following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
        constant_steps (`int`):
            The total number of constant lr steps following a warmup

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    if constant_steps <= 0:
        constant_steps = num_training_steps - num_warmup_steps

    def lr_lambda(current_step):
        # Accelerate sends current_step*num_processes
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step < num_warmup_steps + constant_steps:
            return 1

        return max(
            eps,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps - constant_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_lr_scheduler(name, optimizer, num_warmup_steps, num_training_steps, constant_steps):
    if name != "constant_with_warmup_cosine_decay":
        return get_scheduler(
            name=name,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    # Using custom warmup+constant+decay scheduler
    return get_cosine_schedule_with_warmup_and_decay(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        constant_steps=constant_steps,
    )


@torch.no_grad()
def get_smollm_prompt_embeds(
    tokenizer: AutoTokenizer,
    text_encoder: AutoModelForCausalLM,
    prompts: Union[str, List[str]] = None,
    max_sequence_length: int = 2048,
):
    prompts = [prompts] if isinstance(prompts, str) else prompts
    bot_token_id = 128000  # same as Llama

    if prompts[0] == "":
        bs = len(prompts)
        assert all(p == "" for p in prompts)
        text_input_ids = torch.zeros([bs, 1], dtype=torch.int64, device=text_encoder.device) + bot_token_id
        attention_mask = torch.ones([bs, 1], dtype=torch.int64, device=text_encoder.device)
    else:
        text_inputs = tokenizer(
            prompts,
            padding="longest",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(text_encoder.device)
        attention_mask = text_inputs.attention_mask.to(text_encoder.device)

    if len(prompts) == 1:
        assert (attention_mask == 1).all()

    hidden_states = text_encoder(text_input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states
    # We need a 4096 dim so since we have 2048 we take last 2 layers
    prompt_embeds = torch.concat([hidden_states[-1], hidden_states[-2]], dim=-1)

    return prompt_embeds, hidden_states, attention_mask


def pad_embedding(prompt_embeds, max_tokens):
    # Pads a tensor which is not masked, i.e. the "initial" tensor mask is 1's
    # We extend the tokens to max tokens and provide a mask to differentiate the masked areas
    b, seq_len, dim = prompt_embeds.shape
    padding = torch.zeros(
        (b, max_tokens - seq_len, dim),
        dtype=prompt_embeds.dtype,
        device=prompt_embeds.device,
    )
    attentions_mask = torch.zeros((b, max_tokens), dtype=prompt_embeds.dtype, device=prompt_embeds.device)
    attentions_mask[:, :seq_len] = 1  # original tensor is not masked
    prompt_embeds = torch.concat([prompt_embeds, padding], dim=1)

    return prompt_embeds, attentions_mask


def load_checkpoint(accelerator, args):
    # Load from local checkpoint that sage maker synced to s3 prefix
    global_step = 0
    if args.resume_from_checkpoint != "latest":
        path = os.path.basename(args.resume_from_checkpoint)
    else:
        # Get the most recent checkpoint
        dirs = os.listdir(args.output_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("_")[1]))
        path = dirs[-1] if len(dirs) > 0 else None

    if path is None:
        accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
        args.resume_from_checkpoint = None
    else:
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(args.output_dir, path), map_location="cpu")
        global_step = int(path.split("_")[-1])

    return global_step


class TimestepSampler(abc.ABC):
    """Base class for timestep samplers.

    Timestep samplers are used to sample timesteps for diffusion models.
    They should implement both sample() and sample_for() methods.
    """

    def sample(
        self,
        batch_size: int,
        seq_length: int | None = None,
        device: torch.device = None,
    ) -> torch.Tensor:
        """Sample timesteps for a batch.

        Args:
            batch_size: Number of timesteps to sample
            seq_length: (optional) Length of the sequence being processed
            device: Device to place the samples on

        Returns:
            Tensor of shape (batch_size,) containing timesteps
        """
        raise NotImplementedError

    def sample_for(self, batch: torch.Tensor) -> torch.Tensor:
        """Sample timesteps for a specific batch tensor.

        Args:
            batch: Input tensor of shape (batch_size, seq_length, ...)

        Returns:
            Tensor of shape (batch_size,) containing timesteps
        """
        raise NotImplementedError


class UniformTimestepSampler(TimestepSampler):
    """Samples timesteps uniformly between min_value and max_value (default 0 and 1)."""

    def __init__(self, min_value: float = 0.0, max_value: float = 1.0):
        self.min_value = min_value
        self.max_value = max_value

    def sample(
        self,
        batch_size: int,
        seq_length: int | None = None,
        device: torch.device = None,
    ) -> torch.Tensor:  # noqa: ARG002
        return torch.rand(batch_size, device=device) * (self.max_value - self.min_value) + self.min_value

    def sample_for(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.ndim != 3:
            raise ValueError(f"Batch should have 3 dimensions, got {batch.ndim}")

        batch_size, seq_length, _ = batch.shape
        return self.sample(batch_size, device=batch.device)


class ShiftedLogitNormalTimestepSampler:
    """
    Samples timesteps from a shifted logit-normal distribution,
    where the shift is determined by the sequence length.
    """

    def __init__(self, std: float = 1.0):
        self.std = std

    def sample(self, batch_size: int, seq_length: int, device: torch.device = None) -> torch.Tensor:
        """Sample timesteps for a batch from a shifted logit-normal distribution.

        Args:
            batch_size: Number of timesteps to sample
            seq_length: Length of the sequence being processed, used to determine the shift
            device: Device to place the samples on

        Returns:
            Tensor of shape (batch_size,) containing timesteps sampled from a shifted
            logit-normal distribution, where the shift is determined by seq_length
        """
        shift = self._get_shift_for_sequence_length(seq_length)
        normal_samples = torch.randn((batch_size,), device=device) * self.std + shift
        sigmas = torch.sigmoid(normal_samples)
        return sigmas

    def sample_for(self, batch: torch.Tensor) -> torch.Tensor:
        """Sample timesteps for a specific batch tensor.

        Args:
            batch: Input tensor of shape (batch_size, seq_length, ...)

        Returns:
            Tensor of shape (batch_size,) containing timesteps sampled from a shifted
            logit-normal distribution, where the shift is determined by the sequence length
            of the input batch

        Raises:
            ValueError: If the input batch does not have 3 dimensions
        """
        if batch.ndim != 3:
            raise ValueError(f"Batch should have 3 dimensions, got {batch.ndim}")

        batch_size, seq_length, _ = batch.shape
        return self.sample(batch_size, seq_length, device=batch.device)

    @staticmethod
    def _get_shift_for_sequence_length(
        seq_length: int,
        min_tokens: int = 256,
        max_tokens: int = 4096,
        min_shift: float = 0.5,
        max_shift: float = 1.15,
    ) -> float:
        # Calculate the shift value for a given sequence length using linear interpolation
        # between min_shift and max_shift based on sequence length.
        m = (max_shift - min_shift) / (max_tokens - min_tokens)  # Calculate slope
        b = min_shift - m * min_tokens  # Calculate y-intercept
        shift = m * seq_length + b  # Apply linear equation y = mx + b
        return shift


class ShiftedStretchedLogitNormalTimestepSampler:
    """
    Samples timesteps from a stretched logit-normal distribution,
    where the shift is determined by the sequence length.
    """

    def __init__(self, std: float = 1.0, uniform_prob: float = 0.1):
        self.std = std
        self.shifted_logit_normal_sampler = ShiftedLogitNormalTimestepSampler(std=std)
        self.uniform_sampler = UniformTimestepSampler()
        self.uniform_prob = uniform_prob

    def sample(self, batch_size: int, seq_length: int, device: torch.device = None) -> torch.Tensor:
        # Determine which sampler to use for each batch element
        should_use_uniform = torch.rand(batch_size, device=device) < self.uniform_prob

        # Initialize an empty tensor for the results
        timesteps = torch.empty(batch_size, device=device)

        # Sample from uniform sampler where should_use_uniform is True
        num_uniform = should_use_uniform.sum().item()
        if num_uniform > 0:
            timesteps[should_use_uniform] = self.uniform_sampler.sample(
                batch_size=num_uniform, seq_length=seq_length, device=device
            )

        # Sample from shifted logit-normal sampler where should_use_uniform is False
        should_use_shifted = ~should_use_uniform
        num_shifted = should_use_shifted.sum().item()
        if num_shifted > 0:
            timesteps[should_use_shifted] = self.shifted_logit_normal_sampler.sample(
                batch_size=num_shifted, seq_length=seq_length, device=device
            )
        return timesteps

    def sample_for(self, batch: torch.Tensor) -> torch.Tensor:
        """Sample timesteps for a specific batch tensor.

        Args:
            batch: Input tensor of shape (batch_size, seq_length, ...)

        Returns:
            Tensor of shape (batch_size,) containing timesteps

        Raises:
            ValueError: If the input batch does not have 3 dimensions
        """
        if batch.ndim != 3:
            raise ValueError(f"Batch should have 3 dimensions, got {batch.ndim}")

        batch_size, seq_length, _ = batch.shape
        return self.sample(batch_size=batch_size, seq_length=seq_length, device=batch.device)


def init_training_scheduler():
    return ShiftedStretchedLogitNormalTimestepSampler()


def create_attention_matrix(attention_mask):
    attention_matrix = torch.einsum("bi,bj->bij", attention_mask, attention_mask)

    # convert to 0 - keep, -inf ignore
    attention_matrix = torch.where(
        attention_matrix == 1, 0.0, -torch.inf
    )  # Apply -inf to ignored tokens for nulling softmax score
    return attention_matrix
