#!/usr/bin/env python
"""
Batch Inference Script using FastChat-style evaluation

This script performs batch inference on a test dataset using the trained
Uni-LoRA model, compatible with FastChat's evaluation methodology.

Usage:
    python batch_inference.py --stage2_path <path> --input_file <file> --output_file <file>
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add modeling path
sys.path.insert(0, str(Path(__file__).parent.parent / "modeling"))
from modeling_unilora_moe import apply_unilora_to_qwen_moe

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_model_with_unilora(
    base_model_path: str,
    unilora_params_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """Load base model and apply trained Uni-LoRA parameters."""
    logger.info(f"Loading base model from {base_model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        padding_side="left",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    logger.info("Applying Uni-LoRA architecture...")
    model = apply_unilora_to_qwen_moe(
        model,
        rank=64,
        alpha=16.0,
        use_rank1=True,
    )

    logger.info(f"Loading Uni-LoRA parameters from {unilora_params_path}")
    unilora_params = torch.load(unilora_params_path, map_location="cpu")

    if "unilora_shared_vector" in unilora_params:
        model.unilora_shared_vector.data = unilora_params["unilora_shared_vector"].to(
            model.unilora_shared_vector.device
        )

    loaded_count = 0
    for name, param in model.named_parameters():
        if name in unilora_params:
            param.data = unilora_params[name].to(param.device)
            loaded_count += 1

    logger.info(f"Loaded {loaded_count} projection parameters")
    model.eval()
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> str:
    """Generate response for a single prompt."""
    messages = [{"role": "user", "content": prompt}]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )

    return response


def load_test_data(input_file: str) -> List[Dict[str, Any]]:
    """
    Load test data from file.

    Supports:
    - JSONL format: each line is {"question": "...", "category": "..."}
    - JSON format: list of {"question": "...", "category": "..."}
    - TXT format: one question per line
    """
    data = []
    input_path = Path(input_file)

    if input_path.suffix == ".jsonl":
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    elif input_path.suffix == ".json":
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif input_path.suffix == ".txt":
        with open(input_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if line.strip():
                    data.append(
                        {
                            "question_id": i + 1,
                            "question": line.strip(),
                            "category": "general",
                        }
                    )
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")

    return data


def create_sample_test_file(output_path: str):
    """Create a sample test file for demonstration."""
    sample_questions = [
        {
            "question_id": 1,
            "question": "什么是机器学习？请简要解释。",
            "category": "knowledge",
        },
        {
            "question_id": 2,
            "question": "请解释深度学习和传统机器学习的区别。",
            "category": "knowledge",
        },
        {
            "question_id": 3,
            "question": "写一个Python函数来计算斐波那契数列的第n项。",
            "category": "coding",
        },
        {"question_id": 4, "question": "如何提高代码的可读性？", "category": "coding"},
        {
            "question_id": 5,
            "question": "请用简单的语言解释什么是Transformer架构。",
            "category": "knowledge",
        },
        {
            "question_id": 6,
            "question": "写一首关于春天的短诗。",
            "category": "creative",
        },
        {"question_id": 7, "question": "如何有效地管理时间？", "category": "advice"},
        {
            "question_id": 8,
            "question": "解释MoE（专家混合）模型的工作原理。",
            "category": "knowledge",
        },
        {
            "question_id": 9,
            "question": "请给出三个提高Python代码性能的技巧。",
            "category": "coding",
        },
        {
            "question_id": 10,
            "question": "总结一下大语言模型的主要应用场景。",
            "category": "knowledge",
        },
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sample_questions, f, ensure_ascii=False, indent=2)

    logger.info(f"Created sample test file at {output_path}")
    return sample_questions


def run_batch_inference(
    model,
    tokenizer,
    test_data: List[Dict[str, Any]],
    output_file: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
):
    """Run batch inference on test data."""
    results = []
    total_time = 0

    logger.info(f"Running inference on {len(test_data)} samples...")

    for item in tqdm(test_data, desc="Generating responses"):
        question = item.get("question", item.get("prompt", ""))
        question_id = item.get("question_id", item.get("id", len(results) + 1))
        category = item.get("category", "general")

        start_time = time.time()
        response = generate_response(
            model,
            tokenizer,
            question,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        elapsed = time.time() - start_time
        total_time += elapsed

        result = {
            "question_id": question_id,
            "question": question,
            "category": category,
            "response": response,
            "generation_time": round(elapsed, 2),
        }
        results.append(result)

        # Save intermediate results
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    # Calculate statistics
    avg_time = total_time / len(results) if results else 0
    stats = {
        "total_samples": len(results),
        "total_time": round(total_time, 2),
        "avg_time_per_sample": round(avg_time, 2),
    }

    # Category breakdown
    category_counts = {}
    for r in results:
        cat = r["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
    stats["category_breakdown"] = category_counts

    # Save final results with stats
    final_output = {
        "statistics": stats,
        "results": results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    logger.info(f"Results saved to {output_file}")
    logger.info(f"Statistics: {json.dumps(stats, indent=2)}")

    return results, stats


def main():
    parser = argparse.ArgumentParser(description="Batch Inference with Uni-LoRA")
    parser.add_argument(
        "--base_model",
        type=str,
        default="/root/autodl-tmp/model/Qwen/Qwen1.5-MoE-A2.7B-Chat",
        help="Path to base Qwen1.5-MoE model",
    )
    parser.add_argument(
        "--stage2_path",
        type=str,
        default="../training/output/qwen_moe_unilora_pipeline_0107/stage2",
        help="Path to stage2 output directory",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Input file with test questions (JSON/JSONL/TXT)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./batch_results.json",
        help="Output file for results",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--create_sample",
        action="store_true",
        help="Create a sample test file and run inference on it",
    )

    args = parser.parse_args()

    # Construct path to unilora_params.pt
    unilora_params_path = Path(args.stage2_path) / "unilora_params.pt"
    if not unilora_params_path.exists():
        logger.error(f"unilora_params.pt not found at {unilora_params_path}")
        sys.exit(1)

    # Load model
    model, tokenizer = load_model_with_unilora(
        args.base_model,
        str(unilora_params_path),
    )

    # Load or create test data
    if args.create_sample:
        sample_file = "./sample_test_questions.json"
        test_data = create_sample_test_file(sample_file)
        args.input_file = sample_file
    elif args.input_file:
        test_data = load_test_data(args.input_file)
    else:
        logger.error("Please provide --input_file or use --create_sample")
        sys.exit(1)

    # Run batch inference
    results, stats = run_batch_inference(
        model,
        tokenizer,
        test_data,
        args.output_file,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    print("\n" + "=" * 60)
    print("Batch Inference Complete")
    print("=" * 60)
    print(f"Total samples: {stats['total_samples']}")
    print(f"Total time: {stats['total_time']}s")
    print(f"Avg time per sample: {stats['avg_time_per_sample']}s")
    print(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
