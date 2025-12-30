"""
Training script for Qwen MoE with Uni-LoRA
Supports two-stage training: Stage 1 (freeze router), Stage 2 (freeze adapters, train router)
"""

import argparse
import copy
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

# 添加 modeling 目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "modeling"))
from modeling_unilora_moe import (  # pyright: ignore[reportMissingImports]
    apply_unilora_to_qwen_moe,
    freeze_router,
    freeze_unilora_adapters,
)
from peft import prepare_model_for_kbit_training
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="Qwen/Qwen1.5-MoE-A2.7B-Chat",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={
            "help": "Enable unpickling of arbitrary code in AutoModelForCausalLM"
        },
    )


@dataclass
class DataArguments:
    dataset: str = field(default="alpaca", metadata={"help": "Dataset name or path"})
    dataset_format: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset format: alpaca, chip2, self-instruct, etc."},
    )
    source_max_len: int = field(
        default=1024, metadata={"help": "Maximum source sequence length"}
    )
    target_max_len: int = field(
        default=256, metadata={"help": "Maximum target sequence length"}
    )
    max_train_samples: Optional[int] = field(
        default=None, metadata={"help": "Maximum number of training samples"}
    )
    max_eval_samples: Optional[int] = field(
        default=None, metadata={"help": "Maximum number of evaluation samples"}
    )
    eval_dataset_size: int = field(
        default=1024, metadata={"help": "Size of validation dataset"}
    )


@dataclass
class UniLoRAArguments:
    rank: int = field(default=64, metadata={"help": "Rank of Uni-LoRA adaptation"})
    alpha: float = field(
        default=16.0, metadata={"help": "Scaling factor (alpha) for Uni-LoRA"}
    )
    training_stage: int = field(
        default=1,
        metadata={
            "help": "Training stage: 1 (freeze router, train adapters) or 2 (freeze adapters, train router)"
        },
    )


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding."""
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
    if example.get("input", "") != "":
        prompt_format = ALPACA_PROMPT_DICT["prompt_input"]
    else:
        prompt_format = ALPACA_PROMPT_DICT["prompt_no_input"]
    return {"input": prompt_format.format(**example)}


def local_dataset(dataset_name):
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


def make_data_module(tokenizer: transformers.PreTrainedTokenizer, args):
    """Make dataset and collator for supervised fine-tuning."""

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
        else:
            if os.path.exists(dataset_name):
                try:
                    args.dataset_format = (
                        args.dataset_format if args.dataset_format else "input-output"
                    )
                    full_dataset = local_dataset(dataset_name)
                    return full_dataset
                except Exception as e:
                    raise ValueError(f"Error loading dataset from {dataset_name}: {e}")
            else:
                raise NotImplementedError(
                    f"Dataset {dataset_name} not implemented yet."
                )

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
            dataset = dataset.map(
                lambda x: {
                    "input": "",
                    "output": x["text"],
                }
            )
        elif dataset_format == "input-output":
            pass

        dataset = dataset.remove_columns(
            [
                col
                for col in dataset.column_names["train"]
                if col not in ["input", "output"]
            ]
        )
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
        predict_with_generate=args.predict_with_generate,
    )

    return dict(
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        predict_dataset=eval_dataset if args.do_predict else None,
        data_collator=data_collator,
    )


def print_trainable_parameters(model):
    """Print the number of trainable parameters."""
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            logger.info(
                f"Trainable: {name}, shape: {param.shape}, dtype: {param.dtype}"
            )

    logger.info(
        f"trainable params: {trainable_params:,} || "
        f"all params: {all_param:,} || "
        f"trainable: {100 * trainable_params / all_param:.2f}%"
    )


def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, UniLoRAArguments, TrainingArguments)
    )
    model_args, data_args, unilora_args, training_args = (
        parser.parse_args_into_dataclasses()
    )

    # Merge all args for convenience
    args = argparse.Namespace(
        **vars(model_args),
        **vars(data_args),
        **vars(unilora_args),
        **vars(training_args),
    )

    logger.info(f"Training arguments: {args}")

    # Set seed
    set_seed(training_args.seed)

    # Load model with 4-bit quantization
    logger.info(f"Loading model from {model_args.model_name_or_path}")
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
        torch_dtype=compute_dtype,
        trust_remote_code=model_args.trust_remote_code,
    )

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=training_args.gradient_checkpointing
    )

    # Apply Uni-LoRA to MoE layers
    logger.info("Applying Uni-LoRA to MoE layers...")
    model = apply_unilora_to_qwen_moe(
        model,
        rank=unilora_args.rank,
        alpha=unilora_args.alpha,
    )

    # Load checkpoint if resuming from stage 1
    if unilora_args.training_stage == 2:
        # Try to load from stage 1 checkpoint
        stage1_output_dir = os.path.join(training_args.output_dir, "..", "stage1")
        if os.path.exists(stage1_output_dir):
            # Find best checkpoint
            checkpoints = [
                d for d in os.listdir(stage1_output_dir) if d.startswith("checkpoint-")
            ]
            if checkpoints:
                best_checkpoint = sorted(
                    checkpoints, key=lambda x: int(x.split("-")[1])
                )[-1]
                checkpoint_path = os.path.join(stage1_output_dir, best_checkpoint)
                logger.info(f"Loading Uni-LoRA parameters from {checkpoint_path}")

                # Load Uni-LoRA parameters if they exist
                unilora_params_path = os.path.join(checkpoint_path, "unilora_params.pt")
                if os.path.exists(unilora_params_path):
                    unilora_params = torch.load(unilora_params_path, map_location="cpu")
                    if hasattr(model, "unilora_shared_vector"):
                        model.unilora_shared_vector.data = unilora_params[
                            "unilora_shared_vector"
                        ]
                    logger.info("Loaded Uni-LoRA shared vector from checkpoint")

                # Load model state dict
                model_state_path = os.path.join(checkpoint_path, "pytorch_model.bin")
                if os.path.exists(model_state_path):
                    state_dict = torch.load(model_state_path, map_location="cpu")
                    model.load_state_dict(state_dict, strict=False)
                    logger.info("Loaded model state from checkpoint")

    # Configure training stage
    logger.info(f"Training Stage: {unilora_args.training_stage}")
    if unilora_args.training_stage == 1:
        # Stage 1: Freeze router, train only adapters (v and P_e)
        logger.info("Stage 1: Freezing router, training Uni-LoRA adapters")
        freeze_router(model, freeze=True)
        freeze_unilora_adapters(model, freeze=False)
    elif unilora_args.training_stage == 2:
        # Stage 2: Freeze adapters, train router
        logger.info("Stage 2: Freezing Uni-LoRA adapters, training router")
        freeze_router(model, freeze=False)
        freeze_unilora_adapters(model, freeze=True)
    else:
        raise ValueError(
            f"Invalid training stage: {unilora_args.training_stage}. Must be 1 or 2."
        )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="right",
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )

    model.config.use_cache = False

    # Print trainable parameters
    logger.info("Trainable parameters:")
    print_trainable_parameters(model)

    # Prepare data
    data_module = make_data_module(tokenizer=tokenizer, args=args)

    # Create trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **{k: v for k, v in data_module.items() if k != "predict_dataset"},
    )

    # Training
    if training_args.do_train:
        logger.info("*** Train ***")
        train_result = trainer.train()
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

    # Save model
    if training_args.do_train:
        logger.info("Saving model...")
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)

        # Save Uni-LoRA specific parameters
        if hasattr(model, "unilora_shared_vector"):
            torch.save(
                {
                    "unilora_shared_vector": model.unilora_shared_vector,
                    "rank": unilora_args.rank,
                    "alpha": unilora_args.alpha,
                    "training_stage": unilora_args.training_stage,
                },
                os.path.join(training_args.output_dir, "unilora_params.pt"),
            )


if __name__ == "__main__":
    main()
