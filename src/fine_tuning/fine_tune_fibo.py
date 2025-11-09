import argparse
from datetime import datetime
import itertools
import json
import logging
import os
from pathlib import Path
import random
import subprocess

from PIL import Image
from PIL.ImageOps import exif_transpose
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)
import diffusers
from diffusers import AutoencoderKLWan
from diffusers import BriaFiboPipeline
from diffusers.models.transformers.transformer_bria_fibo import BriaFiboTransformer2DModel

from huggingface_hub import HfFolder
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.fine_tuning.bria_utils import (
    CudaTimerContext,
    create_attention_matrix,
    get_lr_scheduler,
    get_smollm_prompt_embeds,
    init_training_scheduler,
    load_checkpoint,
    pad_embedding,
)
from src.fine_tuning.lora_utils import cast_training_params, set_lora_training

# Set Logger
logger = get_logger(__name__, log_level="INFO")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple example of a training script."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default = "briaai/FIBO",
        required=False,
    )    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR")
        or f"/home/ubuntu/output",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=10,
        help="A seed for reproducible training.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=3000,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,  
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=1501,
        help="Total number of training steps to perform.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1.0,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup",constant_with_warmup_cosine_decay'
        ),
    )
    parser.add_argument(
        "--constant_steps",
        type=int,
        default=-1,
        help=(
            'Amount of constsnt lr steps'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        default=True,
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam and Prodigy optimizers.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam and Prodigy optimizers.",
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay to use.",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-15,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="prodigy",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodigy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=250,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--drop_rate_cfg",
        type=float,
        default=0.,
        help="Rate for Classifier Free Guidance dropping.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default='no',
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_checkpointing",
        type=int,
        default=1,
        required=False,
        help="Path to pretrained ELLA model",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=128, 
    )


    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) containing the training data of instance images (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help=("A folder containing the training data. "),
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing the target image. By "
        "default, the standard Image Dataset maps out 'file_name' "
        "to 'image'.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="caption",
        help="The column of the dataset containing the instance prompt for each image",
    )
    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default="None",
        required=False,
        help="The prompt with identifier specifying the instance, e.g. 'photo of a TOK dog', 'in the style of TOK'",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument(
        "--center_crop",
        default=True,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )

    args = parser.parse_args()
    return args


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        class_prompt,
        class_data_root=None,
        class_num=None,
        size=1024,
        repeats=1,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop

        self.instance_prompt = instance_prompt
        self.custom_instance_prompts = None
        self.class_prompt = class_prompt

        # if --dataset_name is provided or a metadata jsonl file is provided in the local --instance_data directory,
        # we load the training data using load_dataset
        if args.dataset_name is not None:
            try:
                from datasets import load_dataset
            except ImportError:
                raise ImportError(
                    "You are trying to load your data using the datasets library. If you wish to train using custom "
                    "captions please install the datasets library: `pip install datasets`. If you wish to load a "
                    "local folder containing images only, specify --instance_data_dir instead."
                )
            # Downloading and loading a dataset from the hub.
            # See more about loading custom images at
            # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script
            dataset = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                cache_dir=args.cache_dir,
            )
            # Preprocessing the datasets.
            column_names = dataset["train"].column_names

            # 6. Get the column names for input/target.
            if args.image_column is None:
                image_column = column_names[0]
                logger.info(f"image column defaulting to {image_column}")
            else:
                image_column = args.image_column
                if image_column not in column_names:
                    raise ValueError(
                        f"`--image_column` value '{args.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                    )
            instance_images = dataset["train"][image_column]

            if args.caption_column is None:
                logger.info(
                    "No caption column provided, defaulting to instance_prompt for all images. If your dataset "
                    "contains captions/prompts for the images, make sure to specify the "
                    "column as --caption_column"
                )
                self.custom_instance_prompts = None
            else:
                if args.caption_column not in column_names:
                    raise ValueError(
                        f"`--caption_column` value '{args.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                    )
                custom_instance_prompts = dataset["train"][args.caption_column]
                # create final list of captions according to --repeats
                self.custom_instance_prompts = []
                for caption in custom_instance_prompts:
                    self.custom_instance_prompts.extend(itertools.repeat(caption, repeats))
        else:
            self.instance_data_root = Path(instance_data_root)
            if not self.instance_data_root.exists():
                raise ValueError("Instance images root doesn't exists.")

            instance_images = [Image.open(path) for path in list(Path(instance_data_root).iterdir()) if path.is_file()]
            self.custom_instance_prompts = None

        self.instance_images = []
        for img in instance_images:
            self.instance_images.extend(itertools.repeat(img, repeats))

        # Create shared base transform components (used for both instance and class images)
        base_resize = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)
        base_crop = transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size)
        base_to_tensor_normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
            ]
        )

        # Create transform pipeline for class images (used in __getitem__)
        self.image_transforms = transforms.Compose(
            [
                base_resize,
                base_crop,
                *base_to_tensor_normalize.transforms,
            ]
        )

        # Pre-process instance images
        self.pixel_values = []
        for image in self.instance_images:
            image = exif_transpose(image)
            if not image.mode == "RGB":
                image = image.convert("RGB")
            image = base_resize(image)
            image = base_crop(image)
            image = base_to_tensor_normalize(image)
            self.pixel_values.append(image)

        self.num_instance_images = len(self.instance_images)
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
        else:
            self.class_data_root = None

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = self.pixel_values[index % self.num_instance_images]
        example["instance_images"] = instance_image

        if self.custom_instance_prompts:
            caption = self.custom_instance_prompts[index % self.num_instance_images]
            if caption:
                example["instance_prompt"] = caption
            else:
                example["instance_prompt"] = self.instance_prompt

        else:  # custom prompts were provided, but length does not match size of image dataset
            example["instance_prompt"] = self.instance_prompt

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            class_image = exif_transpose(class_image)

            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt"] = self.class_prompt

        return example

def clean_json_caption(caption):
    try:
        caption = json.loads(caption)
        return ujson.dumps(caption, escape_forward_slashes=False)
    except Exception as e:
        print(f"captions must be in json format")

def collate_fn(examples, with_prior_preservation=False):
    pixel_values = [example["instance_images"] for example in examples]
    captions = [example["instance_prompt"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        pixel_values += [example["class_images"] for example in examples]
        captions += [example["class_prompt"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    return pixel_values, captions            

        
def get_accelerator(args, weight_dtype):
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
    )

    # Set huggingface token key if provided
    with accelerator.main_process_first():
        if accelerator.is_local_main_process:
            if os.environ.get("HF_API_TOKEN"):
                HfFolder.save_token(os.environ.get("HF_API_TOKEN"))

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    return accelerator
    
def main(args):
    try:
        cuda_version = torch.version.cuda
        print(f"PyTorch CUDA Version: {cuda_version}")
        result_nvcc = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, check=True)
        print("CUDA Toolkit version (from nvcc):")
        print(result_nvcc.stdout.split('release ')[1].split(',')[0])
    except Exception as e:
        raise Exception("No nvcc found.")

    args = parse_args()
    
    RANK = int(os.environ.get("RANK", 0))

    seed = args.seed + RANK
    set_seed(seed)
    random.seed(seed)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Set accelerator with fsdp/data-parallel
    accelerator = get_accelerator(args, weight_dtype)

    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    TOTAL_BATCH_NO_ACC = args.train_batch_size

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    logger.info(f'TORCH_VERSION {torch.__version__}')
    logger.info(f'DIFFUSERS_VERSION {diffusers.__version__}')

    logger.info('using precompted datasets')
        
    transformer = BriaFiboTransformer2DModel.from_pretrained(
        args. pretrained_model_name_or_path,
        subfolder="transformer",
        low_cpu_mem_usage=False,     # critical: avoid meta tensors
        device_map=None,             # keep on CPU
        torch_dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    )
    transformer = transformer.to(accelerator.device).eval()
    total_num_layers = transformer.config['num_layers'] + transformer.config['num_single_layers']

    logger.info(f'Using precision of {weight_dtype}')
    if args.lora_rank>0:
        logger.info(f'Using LORA with rank {args.lora_rank}')
        transformer.requires_grad_(False)
        transformer.to(dtype=weight_dtype)
        set_lora_training(accelerator, transformer,args.lora_rank)
        # Upcast trainable parameters (LoRA) into fp32 for mixed precision training
        cast_training_params([transformer], dtype=torch.float32)
    else:
        transformer.requires_grad_(True)
        assert transformer.dtype==torch.float32
    
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    get_prompt_embeds_lambda = get_smollm_prompt_embeds
    print("Loading smolLM text encoder")
    
    tokenizer = AutoTokenizer.from_pretrained(args. pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = AutoModelForCausalLM.from_pretrained(
        args. pretrained_model_name_or_path, subfolder="text_encoder",
        torch_dtype=weight_dtype
    ).to(accelerator.device).eval().requires_grad_(False) 
        
    vae_model = AutoencoderKLWan.from_pretrained(args. pretrained_model_name_or_path, subfolder="vae")
    vae_model = vae_model.to(accelerator.device).requires_grad_(False)
    # Read vae config
    vae_config_path = Path(__file__).parent / 'vae_wan.json.out'
    with open(vae_config_path) as f:
        vae_config = json.load(f)
    vae_config["shift_factor"] = torch.tensor(vae_model.config["latents_mean"]).reshape((1,48,1,1)).to(device=accelerator.device)
    vae_config["scaling_factor"] =  1/torch.tensor(vae_model.config["latents_std"]).reshape((1,48,1,1)).to(device=accelerator.device)
    vae_config["compression_rate"] = 16
    vae_config["latent_channels"] = 48
    
    def get_prompt_embedds(prompts):
        prompt_embeddings, text_encoder_layers, attentions_masks = get_prompt_embeds_lambda(
                    tokenizer,
                    text_encoder,
                    prompts=prompts,
                    max_sequence_length=args.max_sequence_length, 
                )
        return prompt_embeddings, text_encoder_layers, attentions_masks
    
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to prodigy"
        )
        args.optimizer = "prodigy"

    if args.lora_rank>0:
        parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    else:
        parameters = transformer.parameters()

    if args.optimizer.lower() == "adamw":
        optimizer_cls = torch.optim.AdamW
        optimizer = optimizer_cls(
            parameters,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    elif args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_cls = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_cls(
            parameters,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )
        
    lr_scheduler = get_lr_scheduler(
            name=args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps
            * accelerator.num_processes,
            num_training_steps=args.max_train_steps
            * accelerator.num_processes,
            constant_steps=args.constant_steps
            * accelerator.num_processes,
                    )
        
    transformer, optimizer, lr_scheduler = accelerator.prepare(
                transformer, optimizer, lr_scheduler
            )

    logger.info("***** Running training *****")

    logger.info(f"diffusers version: {diffusers.__version__}")
        
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0

    if args.resume_from_checkpoint != "no":
        global_step = load_checkpoint(accelerator,args)
    logger.info(f'Using {args.optimizer} with lr: {args.learning_rate}, beta2: {args.adam_beta2}')

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    now = datetime.now()
    times_arr=[]
    # Init dynamic scheduler (resolution dependent)
    height,width = args.resolution, args.resolution
    noise_scheduler = init_training_scheduler()
    
    # encode null prompt ""
    null_conditioning , null_conditioning_layers, _ = get_prompt_embedds([""])
    logger.info('Using empty prompt for null embeddings')
    assert null_conditioning.shape[0]==1
    null_conditioning  = null_conditioning.repeat(args.train_batch_size,1,1).to(dtype=torch.float32)
    null_conditioning_layers  = [layer.repeat(args.train_batch_size,1,1).to(dtype=torch.float32) for layer in null_conditioning_layers]

    vae_scale_factor = 2 ** (len(vae_config['block_out_channels'])-1) if "compression_rate" not in vae_config else vae_config["compression_rate"]
    transformer.train()
    train_loss = 0.0
    generator = torch.Generator(device=accelerator.device).manual_seed(
        seed
    )

    # Dataset and DataLoaders creation:
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        class_prompt=args.class_prompt,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_num=args.num_class_images,
        size=args.resolution,
        repeats=args.repeats,
        center_crop=args.center_crop,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, False),
        num_workers=args.dataloader_num_workers,
    )    
        
    iter_ = iter(train_dataloader)
    for step in range(global_step*args.gradient_accumulation_steps,args.max_train_steps*args.gradient_accumulation_steps):   
        have_batch=False
        while(not have_batch):
            try:
                fetch_time = datetime.now()
                batch = next(iter_)
                fetch_time = datetime.now() - fetch_time
                have_batch=True
            except Exception as e:
                if type(e)==StopIteration:
                    iter_ = iter(train_dataloader)
                    logger.info(f'Rank {RANK} reinit iterator')
                else:
                    raise e

        pixel_values, captions = batch # On precompute we have vae latent and t5 text embedding                     
        latents = vae_model.encode(pixel_values.unsqueeze(dim=2).to(accelerator.device))
        latents = latents.latent_dist.mean 
        # img = vae_model.decode(latents)  # pyright: ignore[reportUnusedVariable]
        latents = latents[:,:,0]
        
        # Get Captions
        text_encoder_time = []
        with CudaTimerContext(text_encoder_time):
            encoder_hidden_states, text_encoder_layers, prompt_attention_mask = get_prompt_embedds(captions)
        text_encoder_time = text_encoder_time[0]
        text_encoder_layers = list(text_encoder_layers)
        #make sure that the number of text encoder layers is equal to the total number of layers in the transformer
        assert len(text_encoder_layers) <= total_num_layers
        text_encoder_layers = text_encoder_layers + [text_encoder_layers[-1]] * (total_num_layers - len(text_encoder_layers))
        null_conditioning_layers = null_conditioning_layers + [null_conditioning_layers[-1]] * (total_num_layers - len(null_conditioning_layers))

        pixel_values = pixel_values.to(device=accelerator.device,dtype=torch.float32)
        encoder_hidden_states = encoder_hidden_states.to(device=accelerator.device,dtype=torch.float32)
        prompt_attention_mask= prompt_attention_mask.to(device=accelerator.device,dtype=torch.float32)
        
        with accelerator.accumulate(transformer):
            latents = (latents - vae_config['shift_factor']) * vae_config['scaling_factor']

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            
            bsz = pixel_values.shape[0]

            seq_len = (height//vae_scale_factor) * (width//vae_scale_factor)

            sigmas = noise_scheduler.sample(bsz, seq_len, device = accelerator.device)
            timesteps = sigmas * 1000
            while len(sigmas.shape) < len(noise.shape):
                sigmas = sigmas.unsqueeze(-1)
            noisy_latents = sigmas * noise + (1.0 - sigmas) * latents
            
            # input for rope positional embeddings for text
            num_text_tokens = encoder_hidden_states.shape[1]
            text_ids = torch.zeros(num_text_tokens, 3).to(device=accelerator.device, dtype=encoder_hidden_states.dtype)
            
            # Sample masks for the edit prompts.
            if args.drop_rate_cfg >0: 
                null_embedding, null_attention_mask = pad_embedding(null_conditioning,max_tokens=num_text_tokens)
                # null embedding for 10% of the images
                random_p = torch.rand(
                    bsz, device=latents.device, generator=generator
                )

                prompt_mask = random_p < args.drop_rate_cfg
                
                prompt_mask = prompt_mask.reshape(bsz, 1, 1)    
                encoder_hidden_states = torch.where(
                    prompt_mask, null_embedding, encoder_hidden_states
                )

                text_encoder_layers = [
                    torch.where(prompt_mask, pad_embedding(null_conditioning_layers[i], max_tokens=num_text_tokens)[0], text_encoder_layers[i])
                    for i in range(len(text_encoder_layers))
                ]


                prompt_mask = prompt_mask.reshape(bsz, 1)    
                prompt_attention_mask = torch.where(
                    prompt_mask, null_attention_mask, prompt_attention_mask
                )
            
            # Get the target for loss depending on the prediction type
            target = noise - latents # V pred
            num_channels_latents = noisy_latents.shape[1]
            latent_height = int(height) // vae_scale_factor
            latent_width = int(width) // vae_scale_factor

            patched_noisy_latents = BriaFiboPipeline._pack_latents_no_patch(noisy_latents, noisy_latents.shape[0], num_channels_latents, latent_height, latent_width)
            patched_latent_image_ids = BriaFiboPipeline._prepare_latent_image_ids(noisy_latents.shape[0], latent_height, latent_width, accelerator.device, noisy_latents.dtype)                
            
            latent_attention_mask = torch.ones([patched_noisy_latents.shape[0],patched_noisy_latents.shape[1]],dtype=latents.dtype,device=latents.device)
            attention_mask = torch.cat([prompt_attention_mask, latent_attention_mask],dim=1)

            # Prepare attention_matrix
            create_mask_time = []
            with CudaTimerContext(create_mask_time):
                attention_mask = create_attention_matrix(attention_mask) # batch, seq => batch, seq, seq            
            create_mask_time = create_mask_time[0]
            
            attention_mask = attention_mask.unsqueeze(dim=1) # for brodoacast to attention heads
            joint_attention_kwargs={'attention_mask':attention_mask}

            forward_time=[]
            with CudaTimerContext(forward_time):
                model_pred = transformer(
                        hidden_states=patched_noisy_latents,
                        timestep=timesteps,
                        encoder_hidden_states=encoder_hidden_states, #[batch,128,height/patch*width/patch]
                        text_encoder_layers=text_encoder_layers,
                        txt_ids=text_ids, 
                        img_ids=patched_latent_image_ids, 
                        return_dict=False,
                        joint_attention_kwargs=joint_attention_kwargs
                    )[0]

                # Un-Patchify latent  (4 -> 1)
                model_pred = BriaFiboPipeline._unpack_latents_no_patch(model_pred, height, width, vae_scale_factor)                
                loss_coeff = WORLD_SIZE / TOTAL_BATCH_NO_ACC
                
                denoising_loss = torch.mean(
                    ((model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                ).sum()
                denoising_loss = loss_coeff * denoising_loss 

                loss = denoising_loss
            forward_time = forward_time[0]

            train_loss += (
                accelerator.gather(loss.detach()).mean().item() / args.gradient_accumulation_steps
            )

            # Backpropagate
            backward_time = []
            with CudaTimerContext(backward_time):
                accelerator.backward(loss)
            backward_time = backward_time[0]

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    parameters, args.max_grad_norm
                )

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
                        
        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1

            if accelerator.is_main_process:
                logger.info(f"train_loss: {train_loss}")
                after = datetime.now() - now
                now = datetime.now()


                times_arr+=[ after.total_seconds()]
                
            train_loss = 0.0

        if (global_step - 1) % args.checkpointing_steps == 0 and (
            global_step - 1
        ) > 0:
            save_path = os.path.join(
                args.output_dir, f"checkpoint_{global_step-1}"
            )

            accelerator.save_state(save_path)
            logger.info(f"Saved state to {save_path}")
            now = datetime.now()

            
        logs = {"step_loss": loss.detach().item()}

        progress_bar.set_postfix(**logs)
        if global_step >= args.max_train_steps:
            break

    # Create the pipeline using the trained modules and save it.
    logger.info("Waiting for everyone :)")
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    os.environ["NCCL_DEBUG"] = "WARN"
    os.environ["FI_PROVIDER"] = "efa"
    os.environ["FI_EFA_USE_DEVICE_RDMA"] = "1"
    os.environ["NCCL_MIN_NCHANNELS"] = "8"
    os.environ["NCCL_NET_GDR_LEVEL"] = "PHB"
    os.environ["NCCL_P2P_LEVEL"] = "NVL"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    os.environ["NCCL_IB_DISABLE"] = "0"

    args = parse_args()
    main(args)
