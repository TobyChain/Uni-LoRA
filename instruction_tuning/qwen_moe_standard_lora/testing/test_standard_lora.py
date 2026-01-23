"""
Testing script for Standard LoRA trained Qwen MoE model.
Loads the base model with PEFT adapter and performs inference.
"""

import argparse
import json
import logging
import os
from typing import List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    base_model_path: str,
    adapter_path: Optional[str] = None,
    bits: int = 16,
    device: str = "cuda",
):
    """Load model and tokenizer, optionally with PEFT adapter."""

    logger.info(f"Loading base model from {base_model_path}")

    # Configure quantization if needed
    quantization_config = None
    if bits == 4:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif bits == 8:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quantization_config,
    )

    # Load PEFT adapter if provided
    if adapter_path and os.path.exists(adapter_path):
        logger.info(f"Loading PEFT adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        logger.info("PEFT adapter loaded successfully")

    model.eval()
    return model, tokenizer


def format_alpaca_prompt(instruction: str, input_text: str = ""):
    """Format prompt using Alpaca format (same as training)."""
    if input_text:
        prompt = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response: "
        )
    else:
        prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n### Response: "
        )
    return prompt


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    use_alpaca_format: bool = True,
):
    """Generate response for a single prompt."""
    
    # Format prompt using Alpaca format if needed (to match training format)
    if use_alpaca_format:
        # Check if prompt is already in Alpaca format
        if "### Instruction:" not in prompt:
            # Treat the entire prompt as instruction
            formatted_prompt = format_alpaca_prompt(prompt)
        else:
            formatted_prompt = prompt
    else:
        formatted_prompt = prompt

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the input prompt from the response
    if use_alpaca_format:
        response = response[len(formatted_prompt) :].strip()
    else:
        response = response[len(prompt) :].strip()
    return response


def batch_generate(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    batch_size: int = 8,
    use_alpaca_format: bool = True,
):
    """Generate responses for multiple prompts using batch processing."""

    results = []
    total_prompts = len(prompts)
    
    # Format prompts using Alpaca format if needed
    if use_alpaca_format:
        formatted_prompts = []
        for prompt in prompts:
            if "### Instruction:" not in prompt:
                formatted_prompts.append(format_alpaca_prompt(prompt))
            else:
                formatted_prompts.append(prompt)
        prompts = formatted_prompts

    for i in range(0, total_prompts, batch_size):
        batch_prompts = prompts[i : i + batch_size]
        logger.info(
            f"Processing batch {i // batch_size + 1}/{(total_prompts + batch_size - 1) // batch_size} (Prompts {i + 1}-{min(i + batch_size, total_prompts)})"
        )

        # Tokenize batch
        inputs = tokenizer(
            batch_prompts, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode batch
        batch_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for j, full_response in enumerate(batch_responses):
            prompt = batch_prompts[j]
            # Remove prompt from response
            response_text = full_response
            if response_text.startswith(prompt):
                response_text = response_text[len(prompt) :].strip()
            
            # Also try to extract just the response part after "### Response:"
            if "### Response:" in response_text:
                response_text = response_text.split("### Response:")[-1].strip()

            results.append(
                {
                    "prompt": prompt,
                    "response": response_text,
                }
            )

    return results


def main():
    parser = argparse.ArgumentParser(description="Test Standard LoRA model")
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Path to the base model",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Path to the PEFT adapter (optional)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--test-prompts-file",
        type=str,
        default=None,
        help="JSON file containing test prompts",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./test_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=16,
        choices=[4, 8, 16],
        help="Quantization bits",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--use-alpaca-format",
        action="store_true",
        default=True,
        help="Use Alpaca format for prompts (matches training format, default: True)",
    )
    parser.add_argument(
        "--no-alpaca-format",
        dest="use_alpaca_format",
        action="store_false",
        help="Disable Alpaca format (use raw prompts)",
    )

    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model_and_tokenizer(
        args.base_model,
        args.adapter_path,
        args.bits,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    if args.interactive:
        # Interactive mode
        print("\n=== Interactive Mode ===")
        print("Type 'quit' or 'exit' to exit.\n")

        while True:
            try:
                prompt = input("You: ").strip()
                if prompt.lower() in ["quit", "exit"]:
                    break
                if not prompt:
                    continue

                response = generate_response(
                    model,
                    tokenizer,
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    use_alpaca_format=args.use_alpaca_format,
                )
                print(f"Model: {response}\n")

            except KeyboardInterrupt:
                break

        print("Goodbye!")

    elif args.test_prompts_file:
        # Batch mode with test prompts file
        logger.info(f"Loading test prompts from {args.test_prompts_file}")

        with open(args.test_prompts_file, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        if isinstance(test_data, list):
            prompts = [
                item.get("prompt", item.get("input", str(item))) for item in test_data
            ]
        else:
            prompts = test_data.get("prompts", [])

        logger.info(f"Loaded {len(prompts)} test prompts")

        results = batch_generate(
            model,
            tokenizer,
            prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            batch_size=args.batch_size,
            use_alpaca_format=args.use_alpaca_format,
        )

        # Save results
        output_file = os.path.join(args.output_dir, "test_results.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"Results saved to {output_file}")

        # Print summary
        print("\n=== Test Results Summary ===")
        for i, result in enumerate(results[:5]):  # Show first 5
            print(f"\n--- Prompt {i + 1} ---")
            print(f"Input: {result['prompt'][:100]}...")
            print(f"Output: {result['response'][:200]}...")

    else:
        # Demo mode with default prompts
        demo_prompts = [
            "请解释什么是中医的阴阳理论？",
            "What is the capital of France?",
            "Write a short poem about spring.",
        ]

        print("\n=== Demo Mode ===")
        # For demo, just use batch size 1 or process all as one batch
        results = batch_generate(
            model,
            tokenizer,
            demo_prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            batch_size=len(demo_prompts),
            use_alpaca_format=args.use_alpaca_format,
        )
        for res in results:
            print(f"\nPrompt: {res['prompt']}")
            print(f"Response: {res['response']}")


if __name__ == "__main__":
    main()
