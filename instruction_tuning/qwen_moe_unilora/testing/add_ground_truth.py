#!/usr/bin/env python3
"""
Add ground truth answers from qa_alpaca_clean.jsonl to comparison.json
"""

import json

# Paths
dataset_path = "/root/autodl-tmp/data/fomat/qa_alpaca_clean.jsonl"
comparison_path = "/root/autodl-tmp/Uni-LoRA/instruction_tuning/qwen_moe_unilora/testing/test_results_tcmqa/comparison.json"
output_path = "/root/autodl-tmp/Uni-LoRA/instruction_tuning/qwen_moe_unilora/testing/test_results_tcmqa/comparison_with_gt.json"

print("Loading comparison data...")
with open(comparison_path, "r", encoding="utf-8") as f:
    comparison_data = json.load(f)

print(f"Loaded {len(comparison_data)} comparison items")

# Create a mapping from instruction to output
print("Building ground truth mapping from dataset...")
instruction_to_output = {}
with open(dataset_path, "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, 1):
        if line.strip():
            try:
                item = json.loads(line)
                instruction = item.get("instruction", "").strip()
                output = item.get("output", "").strip()
                if instruction and output:
                    instruction_to_output[instruction] = output
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON at line {line_num}")
                continue

print(f"Built mapping with {len(instruction_to_output)} instruction-output pairs")

# Add ground truth to comparison data
matches = 0
for item in comparison_data:
    prompt = item["prompt"].strip()
    if prompt in instruction_to_output:
        item["ground_truth_answer"] = instruction_to_output[prompt]
        matches += 1
    else:
        item["ground_truth_answer"] = None
        print(f"Warning: No ground truth found for prompt: {prompt[:50]}...")

print(f"\nMatched {matches}/{len(comparison_data)} prompts with ground truth")

# Save enhanced comparison
print(f"Saving to {output_path}...")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(comparison_data, f, ensure_ascii=False, indent=2)

# Also overwrite the original comparison.json
print(f"Updating original {comparison_path}...")
with open(comparison_path, "w", encoding="utf-8") as f:
    json.dump(comparison_data, f, ensure_ascii=False, indent=2)

print("Done!")
print("\nEnhanced comparison saved to:")
print(f"  - {output_path}")
print(f"  - {comparison_path} (updated)")
