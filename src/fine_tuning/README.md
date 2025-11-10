# FIBO Fine-Tuning Guide

This guide explains how to fine-tune the FIBO model using LoRA (Low-Rank Adaptation) and generate images with the fine-tuned checkpoints.

## Overview

The fine-tuning process uses LoRA to efficiently adapt the FIBO transformer model to your custom dataset. Training saves checkpoints at regular intervals, which can then be used for image generation.

## Dataset Format

Your training dataset should be organized as follows:

```
dataset_directory/
├── image1.jpg
├── image2.jpg
├── ...
└── metadata.csv
```

The `metadata.csv` file should have two columns:
- `file_name`: Name of the image file
- `caption`: JSON-formatted structured prompt describing the image. Consider using constant phrasing (like a trigger word) or "freezing" JSON fields that should have common content across all images (e.g., "style_medium").

Example `metadata.csv`:
```csv
file_name,caption
image1.jpg,"{""short_description"":""A charming bear..."",""objects"":[...]}"
image2.jpg,"{""short_description"":""Another bear..."",""objects"":[...]}"
```

**Note**: The captions must be valid JSON strings. The training script will validate and normalize them automatically.

## Fine-Tuning

### Basic Training Command

```bash
python src/fine_tuning/fine_tune_fibo.py \
  --checkpointing_steps 250 \
  --max_train_steps 1010 \
  --output_dir exmaple_finetune_results \
  --dataset_name briaai/fine_tune_example \
  --lora_rank 64 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing 1
```

### Key Parameters

- `--dataset_name`: Path to your dataset directory (containing images and `metadata.csv`)
- `--output_dir`: Directory where checkpoints will be saved
- `--lora_rank`: LoRA rank (higher = more capacity, default: 128). Common values: 32, 64, 128
- `--train_batch_size`: Batch size per device (default: 1)
- `--gradient_accumulation_steps`: Number of steps to accumulate gradients before updating (default: 4)
- `--max_train_steps`: Total number of training steps
- `--checkpointing_steps`: Save checkpoint every N steps (default: 250)
- `--learning_rate`: Learning rate (default: 1.0 for Prodigy optimizer)
- `--gradient_checkpointing`: Enable gradient checkpointing to save memory (1 = enabled)

### Additional Useful Parameters

- `--caption_column`: Column name in metadata.csv containing captions (default: "caption")
- `--image_column`: Column name in metadata.csv containing image filenames (default: "image")
- `--repeats`: Number of times to repeat each training sample (default: 1)
- `--resume_from_checkpoint`: Resume from a checkpoint path or "latest" (default: "no")
- `--optimizer`: Optimizer type - "prodigy" (default) or "adamw"
- `--mixed_precision`: Mixed precision training - "bf16" (default), "fp16", or "no"


## Checkpoints

During training, checkpoints are saved in the `output_dir` as:
```
output_dir/
├── checkpoint_250/
├── checkpoint_500/
├── checkpoint_750/
└── checkpoint_final/
```

Each checkpoint directory contains:
- Training state (optimizer, scheduler, etc.)
- LoRA weights (saved in a format compatible with `FluxLoraLoaderMixin`)

## Image Generation with Fine-Tuned Checkpoints

After training, you can generate images using your fine-tuned LoRA weights.

### Basic Generation Command

```bash
python src/fine_tuning/generate_with_lora.py \
  --pretrained_model_name_or_path briaai/FIBO \
  --lora_ckpt_path exmaple_finetune_results/checkpoint_final \
  --structered_prompt_path exxample_structured_prompt.json \
  --output_image_path generated_image.png \
  --seed 42
```

### Parameters

- `--pretrained_model_name_or_path`: Base FIBO model path (default: "briaai/FIBO")
- `--lora_ckpt_path`: Path to the checkpoint directory containing LoRA weights (e.g., `checkpoint_final`)
- `--structered_prompt_path`: Path to a JSON file containing the structured prompt
- `--output_image_path`: Where to save the generated image
- `--seed`: Random seed for reproducibility (default: 42)

### Prompt Format

The `--structered_prompt_path` argument should point to a JSON file containing a structured prompt. This prompt should use the same format as the captions you used for training. If your training set included certain fields with repeating content, or a recurring trigger word, make sure to also include those in your generation prompt for best results.

```json
{
  "short_description": "A charming, cartoon-style brown bear...",
  "objects": [...],
  "background_setting": "...",
  "lighting": {...},
  "aesthetics": {...},
  ...
}
```

## Tips

1. **LoRA Rank**: Start with `--lora_rank 64` for most use cases. Increase to 128 for more complex adaptations, or decrease to 32 for simpler tasks.

2. **Training Steps**: Monitor your training loss. Typically 1000-2000 steps is sufficient, but this depends on your dataset size and complexity.

3. **Batch Size**: With `--train_batch_size 1` and `--gradient_accumulation_steps 4`, the effective batch size is 4. Adjust based on your GPU memory.

> **Note:** If your dataset contains images of multiple resolutions (i.e., images are not all the same size), you may encounter issues with `--train_batch_size` greater than 1. In such cases, set the batch size to 1 to avoid shape mismatches.

4. **Memory Optimization**: Enable `--gradient_checkpointing 1` to reduce memory usage at the cost of slightly slower training.

5. **Resuming Training**: Use `--resume_from_checkpoint latest` to resume from the most recent checkpoint, or specify a path like `--resume_from_checkpoint checkpoint_500`.

6. **Multi-GPU Training**: The script supports distributed training. Use `accelerate launch` for multi-GPU setups.

## Environment Variables for Distributed Training

For optimal performance in multi-GPU/distributed training setups (especially on AWS with EFA), you may want to set the following environment variables before running the training script:

```bash
export NCCL_DEBUG=WARN
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export NCCL_MIN_NCHANNELS=8
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_P2P_LEVEL=NVL
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_LAUNCH_BLOCKING=0
export NCCL_IB_DISABLE=0
```

These settings optimize NCCL communication for AWS EFA (Elastic Fabric Adapter) and improve multi-GPU training performance. You can add these to your shell profile or set them before running the training command.

## Troubleshooting

- **Invalid JSON captions**: Ensure all captions in `metadata.csv` are valid JSON. The script will raise an error with details if validation fails.

- **Out of memory**: Reduce `--train_batch_size`, increase `--gradient_accumulation_steps`, or enable `--gradient_checkpointing`.

- **Checkpoint not found**: Verify the checkpoint path exists and contains LoRA weights. The path should point to a `checkpoint_N` directory, not the parent `output_dir`.

