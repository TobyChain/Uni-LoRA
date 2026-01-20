#!/usr/bin/env python
"""
Uni-LoRA Inference Script for Qwen1.5-MoE

This script loads the base Qwen1.5-MoE model and applies the trained Uni-LoRA
adapters for inference.

Usage:
    python inference_unilora.py --stage2_path <path_to_stage2_output>
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

# Add modeling path
sys.path.insert(0, str(Path(__file__).parent.parent / "modeling"))
from modeling_unilora_moe import load_unilora_checkpoint


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# load_model_with_unilora replaced by shared implementation


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
):
    """Generate response for a given prompt."""

    # Format as chat message
    messages = [{"role": "user", "content": prompt}]

    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Generate
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

    # Decode response (skip input tokens)
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )

    return response


def interactive_chat(model, tokenizer):
    """Run an interactive chat session."""
    print("\n" + "=" * 60)
    print("Uni-LoRA Qwen1.5-MoE Interactive Chat")
    print("Type 'quit' or 'exit' to end the session")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("User: ").strip()
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            if not user_input:
                continue

            response = generate_response(model, tokenizer, user_input)
            print(f"\nAssistant: {response}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def main():
    parser = argparse.ArgumentParser(description="Uni-LoRA Inference")
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
        help="Path to stage2 output directory containing unilora_params.pt",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to generate response for (if not provided, starts interactive mode)",
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

    args = parser.parse_args()

    # Construct path to unilora_params.pt
    stage2_path = Path(args.stage2_path)
    unilora_params_path = stage2_path / "unilora_params.pt"

    if not unilora_params_path.exists():
        # Check for adapter_model subdirectory
        adapter_params_path = stage2_path / "adapter_model" / "unilora_params.pt"
        if adapter_params_path.exists():
            unilora_params_path = adapter_params_path
        else:
            logger.error(
                f"unilora_params.pt not found at {unilora_params_path} or {adapter_params_path}"
            )
            sys.exit(1)

    # Load model
    model, tokenizer = load_unilora_checkpoint(
        args.base_model,
        str(unilora_params_path),
    )

    if args.prompt:
        # Single prompt mode
        response = generate_response(
            model,
            tokenizer,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        print(f"\nResponse: {response}")
    else:
        # Interactive mode
        interactive_chat(model, tokenizer)


if __name__ == "__main__":
    main()
