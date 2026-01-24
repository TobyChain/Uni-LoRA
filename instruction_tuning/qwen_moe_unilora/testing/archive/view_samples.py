#!/usr/bin/env python3
"""
Display sample comparisons with ground truth
"""

import json


def display_sample(item, index):
    """Display a single comparison item nicely formatted"""
    print(f"\n{'=' * 80}")
    print(f"Sample {index + 1}")
    print(f"{'=' * 80}")
    print(f"\n📝 Prompt: {item['prompt']}")
    print("\n⏱️  Response Times:")
    print(
        f"   Base: {item['base_model_time']:.2f}s | Trained: {item['trained_model_time']:.2f}s"
    )

    print("\n✅ Ground Truth Answer:")
    gt = item.get("ground_truth_answer", "N/A")
    print(f"   {gt[:200]}{'...' if len(gt) > 200 else ''}")

    print("\n🤖 Base Model Response:")
    base = item["base_model_response"]
    print(f"   {base[:200]}{'...' if len(base) > 200 else ''}")

    print("\n🎯 Trained Model Response:")
    trained = item["trained_model_response"]
    print(f"   {trained[:200]}{'...' if len(trained) > 200 else ''}")


def main():
    comparison_path = "test_results_tcmqa/comparison.json"

    with open(comparison_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} comparison items")

    # Show first 3 samples
    num_samples = min(3, len(data))
    print(f"\nShowing first {num_samples} samples:\n")

    for i in range(num_samples):
        display_sample(data[i], i)

    print(f"\n\n{'=' * 80}")
    print(f"Full comparison available at: {comparison_path}")
    print(f"Total items: {len(data)}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
