import json
import sys
import numpy as np


def analyze(file_path):
    print(f"Analyzing {file_path}...")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("File not found.")
        return

    num_samples = len(data)
    print(f"Total samples: {num_samples}")

    base_times = [item["base_model_time"] for item in data]
    trained_times = [item["trained_model_time"] for item in data]

    base_lens = [len(item["base_model_response"]) for item in data]
    trained_lens = [len(item["trained_model_response"]) for item in data]

    print("\n--- Performance Stats ---")
    print(f"Avg Base Time: {np.mean(base_times):.2f}s +/- {np.std(base_times):.2f}")
    print(
        f"Avg Trained Time: {np.mean(trained_times):.2f}s +/- {np.std(trained_times):.2f}"
    )
    print(f"Speedup: {np.mean(base_times) / np.mean(trained_times):.2f}x (approx)")

    print("\n--- Response Quality Stats (Length) ---")
    print(f"Avg Base Length: {np.mean(base_lens):.1f} chars")
    print(f"Avg Trained Length: {np.mean(trained_lens):.1f} chars")

    # Keyword analysis (simple presence)
    keywords = [
        "基因",
        "精准医疗",
        "代谢",
        "酶",
        "靶点",
        "检测",
        "风险",
        "副作用",
        "疗效",
    ]
    print("\n--- Keyword Presence (Avg count per response) ---")
    for kw in keywords:
        base_cnt = sum([item["base_model_response"].count(kw) for item in data])
        trained_cnt = sum([item["trained_model_response"].count(kw) for item in data])
        print(
            f"'{kw}': Base={base_cnt / num_samples:.1f}, Trained={trained_cnt / num_samples:.1f}"
        )

    # Specific check for structure indicators
    print("\n--- Structure Indicators ---")
    list_markers = ["1.", "2.", "3.", "1、", "2、"]
    base_struct = sum(
        [any(m in item["base_model_response"] for m in list_markers) for item in data]
    )
    trained_struct = sum(
        [
            any(m in item["trained_model_response"] for m in list_markers)
            for item in data
        ]
    )
    print(
        f"Ref containing numbered list: Base={base_struct} ({base_struct / num_samples:.0%}), Trained={trained_struct} ({trained_struct / num_samples:.0%})"
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze(sys.argv[1])
    else:
        analyze("test_results_tcmqa/comparison.json")
