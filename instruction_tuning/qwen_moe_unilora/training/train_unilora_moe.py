"""
Training script for Qwen MoE with Uni-LoRA
Supports two-stage training:
  Stage 1: Train expert adapters (freeze router)
  Stage 2: Train router (freeze adapters)

Based on qlora_unilora.py implementation, adapted for Qwen MoE architecture with DeepSpeed ZeRO-3.
"""

import argparse
import copy
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence

import pandas as pd
import torch
import transformers
from datasets import Dataset, load_dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    set_seed,
)
from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

# Add modeling directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "modeling"))
from modeling_unilora_moe import (  # pyright: ignore[reportMissingImports]
    apply_unilora_to_qwen_moe,
    freeze_router,
    freeze_unilora_adapters,
)

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="Qwen/Qwen1.5-MoE-A2.7B-Chat",
        metadata={"help": "Model identifier or local path."},
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={
            "help": "Enable unpickling of arbitrary code in AutoModelForCausalLM."
        },
    )
    token: Optional[str] = field(
        default=None,
        metadata={"help": "Huggingface auth token."},
    )


@dataclass
class DataArguments:
    dataset: str = field(
        default="alpaca-clean",
        metadata={"help": "Which dataset to finetune on."},
    )
    dataset_format: Optional[str] = field(
        default=None,
        metadata={
            "help": "Dataset format: [alpaca|chip2|self-instruct|hh-rlhf|oasst1|input-output]"
        },
    )
    source_max_len: int = field(
        default=1024,
        metadata={"help": "Maximum source sequence length."},
    )
    target_max_len: int = field(
        default=256,
        metadata={"help": "Maximum target sequence length."},
    )
    eval_dataset_size: int = field(
        default=1024,
        metadata={"help": "Size of validation dataset (examples)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Truncate number of training examples (debug)."},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Truncate number of evaluation examples (debug)."},
    )


@dataclass
class UniLoRAArguments:
    """Uni-LoRA specific arguments"""

    lora_r: int = field(default=64, metadata={"help": "Uni-LoRA rank."})
    lora_alpha: float = field(default=16.0, metadata={"help": "Uni-LoRA alpha."})
    lora_dropout: float = field(default=0.0, metadata={"help": "Uni-LoRA dropout."})

    training_stage: int = field(
        default=1,
        metadata={"help": "Training stage: 1 (train adapters) or 2 (train router)"},
    )
    use_rank1: bool = field(
        default=True,
        metadata={"help": "Use rank-1 shared vector (True) or low-rank matrix (False)"},
    )

    # Vector bank specific learning rate (following qlora_unilora.py)
    learning_rate_vector_bank: float = field(
        default=1e-3,
        metadata={"help": "Learning rate for shared vector bank"},
    )

    # Quantization options (following qlora_unilora.py)
    bits: int = field(
        default=16,
        metadata={"help": "Quantization bits: 4, 8, or 16 (no quantization)"},
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Use double quantization (for 4-bit)"},
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization type: nf4 or fp4"},
    )
    max_memory_MB: int = field(
        default=40000,
        metadata={"help": "Max memory per GPU (MB)"},
    )


@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    """Extended training arguments"""

    cache_dir: Optional[str] = field(default=None)
    train_on_source: bool = field(
        default=False,
        metadata={
            "help": "Whether to train on the input in addition to the target text."
        },
    )
    adam8bit: bool = field(
        default=False,
        metadata={"help": "Use 8-bit Adam optimizer"},
    )

    # Compatibility for older scripts
    evaluation_strategy: Optional[str] = field(
        default=None,
        metadata={"help": "Alias for eval_strategy (compat)."},
    )

    def __post_init__(self):
        # Handle evaluation_strategy alias
        if self.evaluation_strategy is not None:
            current = getattr(self, "eval_strategy", None)
            if current is None or str(current) == "no":
                self.eval_strategy = self.evaluation_strategy

        current = getattr(self, "eval_strategy", None)
        if current is not None and str(current) != "no" and not self.do_eval:
            self.do_eval = True

        # Fix dataset column issue
        self.remove_unused_columns = False

        super().__post_init__()


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding (from qlora_unilora.py)"""
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg


@dataclass
class DataCollatorForCausalLM:
    """Data collator for causal language modeling (from qlora_unilora.py)"""

    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool = False
    predict_with_generate: bool = False

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        sources = [
            f"{self.tokenizer.bos_token}{example['input']}" for example in instances
        ]
        targets = [
            f"{example['output']}{self.tokenizer.eos_token}" for example in instances
        ]

        tokenized_sources = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )

        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources["input_ids"], tokenized_targets["input_ids"]
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor(
                            [IGNORE_INDEX for _ in range(len(tokenized_source))]
                            + copy.deepcopy(tokenized_target)
                        )
                    )
                else:
                    labels.append(
                        torch.tensor(copy.deepcopy(tokenized_source + tokenized_target))
                    )
            else:
                input_ids.append(torch.tensor(tokenized_source))

        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = (
            pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
            if not self.predict_with_generate
            else None
        )

        data_dict = {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict["labels"] = labels
        return data_dict


# Dataset formatting functions (from qlora_unilora.py)
ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    ),
}


def extract_alpaca_dataset(example):
    """Format Alpaca dataset"""
    if example.get("input", "") != "":
        prompt_format = ALPACA_PROMPT_DICT["prompt_input"]
    else:
        prompt_format = ALPACA_PROMPT_DICT["prompt_no_input"]
    return {"input": prompt_format.format(**example)}


def local_dataset(dataset_name):
    """Load local dataset"""
    if dataset_name.endswith(".json") or dataset_name.endswith(".jsonl"):
        full_dataset = Dataset.from_json(path_or_paths=dataset_name)
    elif dataset_name.endswith(".csv"):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name))
    elif dataset_name.endswith(".tsv"):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name, delimiter="\t"))
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_name}")

    split_dataset = full_dataset.train_test_split(test_size=0.1)
    return split_dataset


def make_data_module(tokenizer: transformers.PreTrainedTokenizer, args) -> Dict:
    """Make dataset and collator for supervised fine-tuning (from qlora_unilora.py)"""

    def load_data(dataset_name):
        if dataset_name == "alpaca":
            return load_dataset("tatsu-lab/alpaca")
        elif dataset_name == "alpaca-clean":
            return load_dataset("yahma/alpaca-cleaned")
        elif dataset_name == "chip2":
            return load_dataset("laion/OIG", data_files="unified_chip2.jsonl")
        elif dataset_name == "self-instruct":
            return load_dataset("yizhongw/self_instruct", name="self_instruct")
        elif dataset_name == "hh-rlhf":
            return load_dataset("Anthropic/hh-rlhf")
        elif dataset_name == "oasst1":
            return load_dataset("timdettmers/openassistant-guanaco")
        elif os.path.exists(dataset_name):
            try:
                args.dataset_format = (
                    args.dataset_format if args.dataset_format else "input-output"
                )
                return local_dataset(dataset_name)
            except Exception as e:
                raise ValueError(f"Error loading dataset from {dataset_name}: {e}")
        else:
            raise NotImplementedError(f"Dataset {dataset_name} not implemented yet.")

    def format_dataset(dataset, dataset_format):
        if (
            dataset_format == "alpaca"
            or dataset_format == "alpaca-clean"
            or (dataset_format is None and args.dataset in ["alpaca", "alpaca-clean"])
        ):
            dataset = dataset.map(
                extract_alpaca_dataset, remove_columns=["instruction"]
            )
        elif dataset_format == "chip2" or (
            dataset_format is None and args.dataset == "chip2"
        ):
            dataset = dataset.map(
                lambda x: {
                    "input": x["text"].split("\n<bot>: ")[0].replace("<human>: ", ""),
                    "output": x["text"].split("\n<bot>: ")[1],
                }
            )
        elif dataset_format == "self-instruct" or (
            dataset_format is None and args.dataset == "self-instruct"
        ):
            for old, new in [["prompt", "input"], ["completion", "output"]]:
                dataset = dataset.rename_column(old, new)
        elif dataset_format == "hh-rlhf" or (
            dataset_format is None and args.dataset == "hh-rlhf"
        ):
            dataset = dataset.map(lambda x: {"input": "", "output": x["chosen"]})
        elif dataset_format == "oasst1" or (
            dataset_format is None and args.dataset == "oasst1"
        ):
            dataset = dataset.map(lambda x: {"input": "", "output": x["text"]})
        elif dataset_format == "input-output":
            pass

        # Remove unused columns
        try:
            dataset = dataset.remove_columns(
                [
                    col
                    for col in dataset.column_names["train"]
                    if col not in ["input", "output"]
                ]
            )
        except:  # For single dataset (not DatasetDict)  # noqa: E722
            pass

        return dataset

    dataset = load_data(args.dataset)
    dataset = format_dataset(dataset, args.dataset_format)

    # Split train/eval
    if args.do_eval or args.do_predict:
        if "eval" in dataset:
            eval_dataset = dataset["eval"]
        else:
            logger.info("Splitting train dataset in train and validation")
            dataset = dataset["train"].train_test_split(
                test_size=args.eval_dataset_size, shuffle=True, seed=42
            )
            eval_dataset = dataset["test"]
        if (
            args.max_eval_samples is not None
            and len(eval_dataset) > args.max_eval_samples
        ):
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        if args.group_by_length:
            eval_dataset = eval_dataset.map(
                lambda x: {"length": len(x["input"]) + len(x["output"])}
            )

    if args.do_train:
        train_dataset = dataset["train"]
        if (
            args.max_train_samples is not None
            and len(train_dataset) > args.max_train_samples
        ):
            train_dataset = train_dataset.select(range(args.max_train_samples))
        if args.group_by_length:
            train_dataset = train_dataset.map(
                lambda x: {"length": len(x["input"]) + len(x["output"])}
            )

    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        train_on_source=args.train_on_source,
        predict_with_generate=args.predict_with_generate,
    )

    return dict(
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        predict_dataset=eval_dataset if args.do_predict else None,
        data_collator=data_collator,
    )


def create_optimizer(model, args) -> torch.optim.Optimizer:
    """
    Create optimizer with separate learning rates for vector bank and other parameters.
    Based on qlora_unilora.py implementation.
    """
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]

    vector_bank_parameters = [
        name for name, _ in model.named_parameters() if "unilora_shared_vector" in name
    ]

    projection_parameters = [
        name
        for name, _ in model.named_parameters()
        if "projection" in name or "projections" in name
    ]

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if n in decay_parameters
                and n not in vector_bank_parameters
                and n not in projection_parameters
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if n not in decay_parameters
                and n not in vector_bank_parameters
                and n not in projection_parameters
            ],
            "weight_decay": 0.0,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if n in vector_bank_parameters
            ],
            "lr": args.learning_rate_vector_bank,
            "weight_decay": 0.0,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if n in projection_parameters
            ],
            "lr": args.learning_rate,
            "weight_decay": 0.0,
        },
    ]

    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(args)
    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    # Handle 8-bit Adam if requested
    if args.adam8bit:
        try:
            import bitsandbytes as bnb

            manager = bnb.optim.GlobalOptimManager.get_instance()
            for module in model.modules():
                if isinstance(module, torch.nn.Embedding):
                    skipped = sum(
                        {p.data_ptr(): p.numel() for p in module.parameters()}.values()
                    )
                    logger.info(f"Skipped {module}: {skipped / 2**20}M params")
                    manager.register_module_override(
                        module, "weight", {"optim_bits": 32}
                    )
                    logger.debug(f"bitsandbytes: will optimize {module} in fp32")
        except ImportError:
            logger.warning("bitsandbytes not available, ignoring adam8bit flag")

    return optimizer


class SaveUniLoRACallback(transformers.TrainerCallback):
    """
    Callback to save Uni-LoRA specific parameters.
    Based on SavePeftModelCallback from qlora_unilora.py.
    """

    def save_model(self, args, state, kwargs):
        logger.info("Saving Uni-LoRA checkpoint...")
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(
                state.best_model_checkpoint, "adapter_model"
            )
        else:
            checkpoint_folder = os.path.join(
                args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            )

        model = kwargs["model"]

        # Save Uni-LoRA specific parameters
        if hasattr(model, "unilora_shared_vector"):
            unilora_params = {
                "unilora_shared_vector": model.unilora_shared_vector.data,
            }
            # Collect all projection parameters
            for name, param in model.named_parameters():
                if "projection" in name or "projections" in name:
                    unilora_params[name] = param.data

            os.makedirs(checkpoint_folder, exist_ok=True)
            torch.save(
                unilora_params, os.path.join(checkpoint_folder, "unilora_params.pt")
            )
            logger.info(f"Saved Uni-LoRA parameters to {checkpoint_folder}")

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, "a"):
                os.utime(fname, times)

        touch(os.path.join(args.output_dir, "completed"))
        self.save_model(args, state, kwargs)


def print_trainable_parameters(args, model):
    """
    Print the number of trainable parameters in the model.
    From qlora_unilora.py.
    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            num = param.numel()
            logger.info(
                f"Trainable: {name}, dtype: {param.dtype}, shape: {param.shape}, params: {num}"
            )
            trainable_params += num

    # Adjust for quantization
    if args.bits == 4:
        trainable_params /= 2

    logger.info(
        f"trainable params: {trainable_params:,} || "
        f"all params: {all_param:,} || "
        f"trainable: {100 * trainable_params / all_param:.2f}%"
    )


def get_last_checkpoint(checkpoint_dir):
    """Get the last checkpoint from a directory (from qlora_unilora.py)"""
    if os.path.isdir(checkpoint_dir):
        is_completed = os.path.exists(os.path.join(checkpoint_dir, "completed"))
        if is_completed:
            return None, True  # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if os.path.isdir(
                os.path.join(checkpoint_dir, filename)
            ) and filename.startswith("checkpoint"):
                max_step = max(max_step, int(filename.replace("checkpoint-", "")))
        if max_step == 0:
            return None, is_completed  # training started, but no checkpoint
        checkpoint_dir = os.path.join(checkpoint_dir, f"checkpoint-{max_step}")
        logger.info(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed  # checkpoint found!
    return None, False  # first training


def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, UniLoRAArguments, TrainingArguments)
    )
    model_args, data_args, unilora_args, training_args, remaining = (
        parser.parse_args_into_dataclasses(return_remaining_strings=True)
    )

    if remaining:
        logger.warning(f"Unrecognized arguments (will be ignored): {remaining}")

    # Merge args for convenience
    args = argparse.Namespace(
        **vars(model_args),
        **vars(data_args),
        **vars(unilora_args),
        **vars(training_args),
    )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    logger.info(f"Training/evaluation parameters:\n{training_args}")
    logger.info(
        f"Uni-LoRA parameters: rank={unilora_args.lora_r}, alpha={unilora_args.lora_alpha}, stage={unilora_args.training_stage}"
    )

    # DeepSpeed ZeRO-3 handling
    ds_zero_stage = None
    if getattr(training_args, "deepspeed", None):
        ds_path = training_args.deepspeed
        logger.info(f"DeepSpeed config path: {ds_path}")

        # Resolve relative path
        if not os.path.exists(ds_path):
            script_dir = Path(__file__).parent
            candidate = script_dir / ds_path
            if candidate.exists():
                ds_path = str(candidate)
                logger.info(f"Resolved DeepSpeed config path to: {ds_path}")
                training_args.deepspeed = ds_path

        try:
            with open(ds_path, encoding="utf-8") as f:
                ds_cfg = json.load(f)
            zero_cfg = (
                ds_cfg.get("zero_optimization", {}) if isinstance(ds_cfg, dict) else {}
            )
            try:
                ds_zero_stage = (
                    int(zero_cfg.get("stage")) if "stage" in zero_cfg else None
                )
            except Exception:
                ds_zero_stage = None
            logger.info(f"Detected DeepSpeed ZeRO stage: {ds_zero_stage}")
        except FileNotFoundError:
            logger.error(f"DeepSpeed config file not found at: {ds_path}")
        except Exception as e:
            logger.error(f"Error parsing DeepSpeed config: {e}")

    # ZeRO-3 initialization (following qlora_unilora.py pattern)
    if ds_zero_stage == 3 and getattr(training_args, "deepspeed", None):
        try:
            from transformers.integrations.deepspeed import HfDeepSpeedConfig

            # Initialize DeepSpeed config (no need to keep reference)
            _ = HfDeepSpeedConfig(training_args.deepspeed)
            logger.info(
                "Enabled ZeRO-3 partitioned initialization via HfDeepSpeedConfig."
            )
        except Exception as e:
            logger.warning(f"Failed to enable HfDeepSpeedConfig for ZeRO-3: {e}")

    # Set seed
    set_seed(training_args.seed)

    # Check for existing checkpoint
    checkpoint_dir, completed_training = get_last_checkpoint(training_args.output_dir)
    if completed_training:
        logger.info("Detected that training was already completed!")
        return

    # Load model
    logger.info(f"Loading model from {model_args.model_name_or_path}")
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    # Quantization config (following qlora_unilora.py)
    quantization_config = None
    if unilora_args.bits == 4:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=unilora_args.double_quant,
            bnb_4bit_quant_type=unilora_args.quant_type,
        )
        logger.info("Using 4-bit quantization with NF4")
    elif unilora_args.bits == 8:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        logger.info("Using 8-bit quantization")
    else:
        logger.info("Using full precision (bf16/fp16)")

    # Device map for ZeRO
    device_map = None
    max_memory = None
    if ds_zero_stage == 3:
        # ZeRO-3 handles device placement
        device_map = None
    elif os.environ.get("LOCAL_RANK") is not None:
        # Distributed but not ZeRO-3
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device_map = {"": local_rank}
        max_memory = {local_rank: f"{unilora_args.max_memory_MB}MB"}
    else:
        # Single GPU or auto
        device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        device_map=device_map,
        max_memory=max_memory,
        quantization_config=quantization_config,
        dtype=compute_dtype,
        trust_remote_code=model_args.trust_remote_code,
        token=model_args.token,
        low_cpu_mem_usage=True,
    )

    # Freeze base model (critical for Uni-LoRA)
    logger.info("Freezing base model parameters...")
    for param in model.parameters():
        param.requires_grad = False

    # Apply Uni-LoRA
    logger.info("Applying Uni-LoRA to MoE layers...")
    model = apply_unilora_to_qwen_moe(
        model,
        rank=unilora_args.lora_r,
        alpha=unilora_args.lora_alpha,
        use_rank1=unilora_args.use_rank1,
    )

    # Load checkpoint if resuming from Stage 1
    if unilora_args.training_stage == 2:
        stage1_output_dir = os.path.join(training_args.output_dir, "..", "stage1")
        if os.path.exists(stage1_output_dir):
            stage1_checkpoint, _ = get_last_checkpoint(stage1_output_dir)
            if stage1_checkpoint:
                logger.info(f"Loading Uni-LoRA parameters from {stage1_checkpoint}")
                unilora_params_path = os.path.join(
                    stage1_checkpoint, "adapter_model", "unilora_params.pt"
                )
                if os.path.exists(unilora_params_path):
                    unilora_params = torch.load(unilora_params_path, map_location="cpu")
                    if "unilora_shared_vector" in unilora_params:
                        model.unilora_shared_vector.data = unilora_params[
                            "unilora_shared_vector"
                        ]
                    # Load projection parameters
                    for name, param in model.named_parameters():
                        if name in unilora_params:
                            param.data = unilora_params[name]
                    logger.info("Loaded Uni-LoRA parameters from Stage 1")

    # Configure training stage
    logger.info(f"Training Stage: {unilora_args.training_stage}")
    if unilora_args.training_stage == 1:
        logger.info("Stage 1: Freezing router, training Uni-LoRA adapters")
        freeze_router(model, freeze=True)
        freeze_unilora_adapters(model, freeze=False)
    elif unilora_args.training_stage == 2:
        logger.info("Stage 2: Freezing Uni-LoRA adapters, training router")
        freeze_router(model, freeze=False)
        freeze_unilora_adapters(model, freeze=True)
    else:
        raise ValueError(
            f"Invalid training stage: {unilora_args.training_stage}. Must be 1 or 2."
        )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        token=model_args.token,
        padding_side="right",
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )

    # Special token handling for Qwen models
    if "qwen" in model_args.model_name_or_path.lower():
        logger.info("Detected Qwen model, configuring special tokens")
        if tokenizer.eos_token is None:
            tokenizer.eos_token = tokenizer.decode([model.config.eos_token_id])
        if tokenizer.bos_token is None:
            tokenizer.bos_token = (
                tokenizer.decode([model.config.bos_token_id])
                if hasattr(model.config, "bos_token_id")
                else ""
            )

    model.config.use_cache = False

    # Print trainable parameters
    logger.info("Trainable parameters:")
    print_trainable_parameters(args, model)

    # Prepare data
    data_module = make_data_module(tokenizer=tokenizer, args=args)

    # Create optimizer with separate learning rates
    optimizer = create_optimizer(model, args)

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        optimizers=(optimizer, None),
        **{k: v for k, v in data_module.items() if k != "predict_dataset"},
    )

    # Add callback
    trainer.add_callback(SaveUniLoRACallback)

    # Training
    if training_args.do_train:
        logger.info("*** Train ***")
        train_result = trainer.train(resume_from_checkpoint=checkpoint_dir)
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Save final model
    if training_args.do_train:
        logger.info("Saving final model...")
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)

        # Save Uni-LoRA specific parameters
        if hasattr(model, "unilora_shared_vector"):
            unilora_params = {"unilora_shared_vector": model.unilora_shared_vector.data}
            for name, param in model.named_parameters():
                if "projection" in name or "projections" in name:
                    unilora_params[name] = param.data

            torch.save(
                unilora_params,
                os.path.join(training_args.output_dir, "unilora_params.pt"),
            )
            logger.info("Saved Uni-LoRA parameters")


if __name__ == "__main__":
    main()
