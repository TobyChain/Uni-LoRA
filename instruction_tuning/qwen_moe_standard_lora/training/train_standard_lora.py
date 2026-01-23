"""
Training script for Qwen MoE with Standard LoRA (PEFT)
Uses HuggingFace PEFT library for standard LoRA training.
"""

import argparse
import copy
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import pandas as pd
import torch
import transformers
from datasets import load_dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments as HfTrainingArguments,
    set_seed,
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
        default=900,
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
class LoRAArguments:
    """Standard LoRA specific arguments"""

    lora_r: int = field(default=64, metadata={"help": "LoRA rank."})
    lora_alpha: float = field(default=16.0, metadata={"help": "LoRA alpha."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout."})

    # Target modules for LoRA
    target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma separated list of target modules for LoRA."},
    )

    # Flash Attention 2
    use_flash_attention_2: bool = field(
        default=True,
        metadata={"help": "Use Flash Attention 2 for memory-efficient attention"},
    )

    # Fused AdamW optimizer
    use_fused_adamw: bool = field(
        default=True,
        metadata={"help": "Use fused AdamW optimizer for improved training speed"},
    )

    # Quantization settings
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
class TrainingArguments(HfTrainingArguments):
    """Extended training arguments"""

    cache_dir: Optional[str] = field(default=None)
    train_on_source: bool = field(
        default=False,
        metadata={
            "help": "Whether to train on the input in addition to the target text."
        },
    )

    evaluation_strategy: Optional[str] = field(
        default=None,
        metadata={"help": "Alias for eval_strategy (compat)."},
    )

    def __post_init__(self):
        if self.evaluation_strategy is not None:
            current = getattr(self, "eval_strategy", None)
            if current is None or str(current) == "no":
                self.eval_strategy = self.evaluation_strategy

        current = getattr(self, "eval_strategy", None)
        if current is not None and str(current) != "no" and not self.do_eval:
            self.do_eval = True

        self.remove_unused_columns = False

        super().__post_init__()


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding"""
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
    """Data collator for causal language modeling"""

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
        full_dataset = load_dataset("json", data_files=dataset_name, split="train")
    elif dataset_name.endswith(".csv"):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name))
    elif dataset_name.endswith(".tsv"):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name, delimiter="\t"))
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_name}")

    split_dataset = full_dataset.train_test_split(test_size=0.1)
    return split_dataset


def make_data_module(tokenizer: transformers.PreTrainedTokenizer, args) -> Dict:
    """Make dataset and collator for supervised fine-tuning"""

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

        try:
            dataset = dataset.remove_columns(
                [
                    col
                    for col in dataset.column_names["train"]
                    if col not in ["input", "output"]
                ]
            )
        except:  # noqa: E722
            pass

        return dataset

    dataset = load_data(args.dataset)
    dataset = format_dataset(dataset, args.dataset_format)

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
        predict_with_generate=getattr(args, "predict_with_generate", False),
    )

    return dict(
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        predict_dataset=eval_dataset if args.do_predict else None,
        data_collator=data_collator,
    )


def print_trainable_parameters(model):
    """Print the number of trainable parameters in the model."""
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

    logger.info(
        f"trainable params: {trainable_params:,} || "
        f"all params: {all_param:,} || "
        f"trainable: {100 * trainable_params / all_param:.2f}%"
    )


def get_last_checkpoint(checkpoint_dir):
    """Get the last checkpoint from a directory"""
    if os.path.isdir(checkpoint_dir):
        is_completed = os.path.exists(os.path.join(checkpoint_dir, "completed"))
        if is_completed:
            return None, True
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if os.path.isdir(
                os.path.join(checkpoint_dir, filename)
            ) and filename.startswith("checkpoint"):
                max_step = max(max_step, int(filename.replace("checkpoint-", "")))
        if max_step == 0:
            return None, is_completed
        checkpoint_dir = os.path.join(checkpoint_dir, f"checkpoint-{max_step}")
        logger.info(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed
    return None, False


def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, LoRAArguments, TrainingArguments)
    )
    model_args, data_args, lora_args, training_args, remaining = (
        parser.parse_args_into_dataclasses(return_remaining_strings=True)
    )

    if remaining:
        logger.warning(f"Unrecognized arguments (will be ignored): {remaining}")

    args = argparse.Namespace(
        **vars(model_args),
        **vars(data_args),
        **vars(lora_args),
        **vars(training_args),
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    logger.info(f"Training/evaluation parameters:\n{training_args}")
    logger.info(
        f"LoRA parameters: rank={lora_args.lora_r}, alpha={lora_args.lora_alpha}, dropout={lora_args.lora_dropout}"
    )

    set_seed(training_args.seed)

    checkpoint_dir, completed_training = get_last_checkpoint(training_args.output_dir)
    if completed_training:
        logger.info("Detected that training was already completed!")
        return

    logger.info(f"Loading model from {model_args.model_name_or_path}")
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    quantization_config = None
    if lora_args.bits == 4:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=lora_args.double_quant,
            bnb_4bit_quant_type=lora_args.quant_type,
        )
        logger.info("Using 4-bit quantization with NF4")
    elif lora_args.bits == 8:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        logger.info("Using 8-bit quantization")
    else:
        logger.info("Using full precision (bf16/fp16)")

    # Device map configuration
    device_map = "auto"
    if os.environ.get("LOCAL_RANK") is not None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device_map = {"": local_rank}

    # Configure Flash Attention 2
    # PyTorch 2.0+ has built-in SDPA with Flash Attention support
    attn_implementation = None
    if lora_args.use_flash_attention_2:
        # Check PyTorch version for SDPA support
        torch_version = tuple(
            int(x) for x in torch.__version__.split("+")[0].split(".")[:2]
        )
        if torch_version >= (2, 0):
            attn_implementation = "sdpa"  # Use PyTorch's built-in SDPA
            logger.info(
                f"Using PyTorch SDPA (built-in Flash Attention) - PyTorch {torch.__version__}"
            )
        else:
            # Fallback to external flash-attn if available
            try:
                import flash_attn

                attn_implementation = "flash_attention_2"
                logger.info(
                    f"Flash Attention 2 enabled (version: {flash_attn.__version__})"
                )
            except ImportError:
                logger.warning(
                    "Flash Attention not available. Using default attention."
                )
                attn_implementation = None

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        device_map=device_map,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
        trust_remote_code=model_args.trust_remote_code,
        token=model_args.token,
        low_cpu_mem_usage=True,
        attn_implementation=attn_implementation,
    )

    # Prepare model for training with quantization
    if lora_args.bits in [4, 8]:
        logger.info("Preparing model for k-bit training...")
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
        )

    # Parse target modules
    target_modules = [m.strip() for m in lora_args.target_modules.split(",")]
    logger.info(f"Target modules for LoRA: {target_modules}")

    # Configure LoRA
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=int(lora_args.lora_alpha),
        target_modules=target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    logger.info("Applying LoRA to model...")
    model = get_peft_model(model, lora_config)

    # Load tokenizer
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

    logger.info("Trainable parameters:")
    print_trainable_parameters(model)
    model.print_trainable_parameters()

    if training_args.gradient_checkpointing:
        logger.info("Enabling gradient checkpointing for memory optimization...")
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    data_module = make_data_module(tokenizer=tokenizer, args=args)

    # Configure optimizer with Fused AdamW if available
    optimizers = (None, None)  # Default: let Trainer handle optimizer
    if lora_args.use_fused_adamw:
        try:
            from torch.optim import AdamW

            # Check if fused implementation is available (requires CUDA)
            if torch.cuda.is_available():
                trainable_params = [p for p in model.parameters() if p.requires_grad]
                optimizer = AdamW(
                    trainable_params,
                    lr=training_args.learning_rate,
                    betas=(training_args.adam_beta1, training_args.adam_beta2),
                    eps=training_args.adam_epsilon,
                    weight_decay=training_args.weight_decay,
                    fused=True,  # Enable fused implementation
                )
                optimizers = (optimizer, None)
                logger.info("Fused AdamW optimizer enabled for improved training speed")
            else:
                logger.warning("Fused AdamW requires CUDA. Using default optimizer.")
        except Exception as e:
            logger.warning(
                f"Failed to create fused AdamW optimizer: {e}. Using default."
            )

    trainer = Trainer(
        model=model,
        args=training_args,
        optimizers=optimizers,
        **{k: v for k, v in data_module.items() if k != "predict_dataset"},
    )

    if training_args.do_train:
        logger.info("*** Train ***")
        train_result = trainer.train(resume_from_checkpoint=checkpoint_dir)
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_train:
        logger.info("Saving final model...")
        # Save the PEFT model
        model.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)

        # Create a completed marker
        with open(os.path.join(training_args.output_dir, "completed"), "w") as f:
            f.write("Training completed successfully.\n")

        logger.info(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()
