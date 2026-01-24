import json
import os
import jieba
from rouge_score import rouge_scorer
import numpy as np
from tabulate import tabulate
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# 路径配置
BASE_DIR = "/root/autodl-tmp/Uni-LoRA/instruction_tuning/qwen_moe_unilora/testing"
DATASET_FILE = "/root/autodl-tmp/data/fomat/qa_alpaca_dedup_exact.jsonl"
PROMPTS_FILE = os.path.join(BASE_DIR, "test_prompts_tcmqa.json")

RESULTS_FILES = {
    "Base Model": os.path.join(
        BASE_DIR, "test_results_combine/base_only/base_model_results.json"
    ),
    "Exp1 (Joint)": os.path.join(
        BASE_DIR,
        "test_results_combine/exp1_joint_only_fixed/trained_model_results.json",
    ),
    "Exp2 (Attn)": os.path.join(
        BASE_DIR, "test_results_combine/exp2_attn_only/trained_model_results.json"
    ),
    "Exp3 (Full)": os.path.join(
        BASE_DIR, "test_results_combine/exp3_full_combined/trained_model_results.json"
    ),
}


def load_json(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def tokenize_chinese(text):
    """使用 jieba 进行分词"""
    return " ".join(jieba.cut(text))


def calculate_detailed_metrics(hyps, refs):
    """计算详细的 BLEU 1-4 和 ROUGE P/R/F1 分数"""
    r_scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=False
    )
    smooth = SmoothingFunction().method1

    results = {
        "BLEU-1": [],
        "BLEU-2": [],
        "BLEU-3": [],
        "BLEU-4": [],
        "R1-P": [],
        "R1-R": [],
        "R1-F1": [],
        "R2-P": [],
        "R2-R": [],
        "R2-F1": [],
        "RL-P": [],
        "RL-R": [],
        "RL-F1": [],
    }

    for hyp, ref in zip(hyps, refs):
        hyp_tok = tokenize_chinese(hyp).split()
        ref_tok = [tokenize_chinese(ref).split()]

        # BLEU 1-4 using nltk (cumulative)
        results["BLEU-1"].append(
            sentence_bleu(
                ref_tok, hyp_tok, weights=(1, 0, 0, 0), smoothing_function=smooth
            )
            * 100
        )
        results["BLEU-2"].append(
            sentence_bleu(
                ref_tok, hyp_tok, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth
            )
            * 100
        )
        results["BLEU-3"].append(
            sentence_bleu(
                ref_tok,
                hyp_tok,
                weights=(0.33, 0.33, 0.33, 0),
                smoothing_function=smooth,
            )
            * 100
        )
        results["BLEU-4"].append(
            sentence_bleu(
                ref_tok,
                hyp_tok,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smooth,
            )
            * 100
        )

        # ROUGE
        hyp_str = " ".join(hyp_tok)
        ref_str = " ".join(ref_tok[0])
        rs = r_scorer.score(ref_str, hyp_str)

        results["R1-P"].append(rs["rouge1"].precision * 100)
        results["R1-R"].append(rs["rouge1"].recall * 100)
        results["R1-F1"].append(rs["rouge1"].fmeasure * 100)

        results["R2-P"].append(rs["rouge2"].precision * 100)
        results["R2-R"].append(rs["rouge2"].recall * 100)
        results["R2-F1"].append(rs["rouge2"].fmeasure * 100)

        results["RL-P"].append(rs["rougeL"].precision * 100)
        results["RL-R"].append(rs["rougeL"].recall * 100)
        results["RL-F1"].append(rs["rougeL"].fmeasure * 100)

    return {k: np.mean(v) for k, v in results.items()}


def main():
    prompts_data = load_json(PROMPTS_FILE)
    prompts_list = [item.get("prompt", "").strip() for item in prompts_data]

    true_responses = {}
    if os.path.exists(DATASET_FILE):
        with open(DATASET_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    p = (
                        item.get("instruction", "")
                        or item.get("prompt", "")
                        or item.get("input", "")
                    ).strip()
                    o = item.get("output", "") or item.get("response", "")
                    if p:
                        true_responses[p] = o
                except:
                    pass

    all_metrics = []

    for model_name, file_path in RESULTS_FILES.items():
        print(f"正在分析 {model_name}...")
        results = load_json(file_path)
        hyps, refs = [], []
        results_map = {
            item.get("prompt", "").strip(): item.get("response", "") for item in results
        }

        for p in prompts_list:
            if p in results_map and p in true_responses:
                hyps.append(results_map[p])
                refs.append(true_responses[p])

        if hyps:
            m = calculate_detailed_metrics(hyps, refs)
            m["Model"] = model_name
            all_metrics.append(m)

    # 显示结果表格
    headers = ["Model", "B-1", "B-2", "B-3", "B-4", "R1-F1", "R2-F1", "RL-F1"]
    table_data = [
        [
            m["Model"],
            f"{m['BLEU-1']:.2f}",
            f"{m['BLEU-2']:.2f}",
            f"{m['BLEU-3']:.2f}",
            f"{m['BLEU-4']:.2f}",
            f"{m['R1-F1']:.2f}",
            f"{m['R2-F1']:.2f}",
            f"{m['RL-F1']:.2f}",
        ]
        for m in all_metrics
    ]

    print("\n### 综合指标汇总 (BLEU 累积 & ROUGE F1)")
    print(tabulate(table_data, headers=headers, tablefmt="github"))

    # 详细 ROUGE P/R 对比
    print("\n### ROUGE 详细对比 (Precision/Recall/F1)")
    rouge_headers = [
        "Model",
        "R1-P",
        "R1-R",
        "R1-F1",
        "R2-P",
        "R2-R",
        "R2-F1",
        "RL-P",
        "RL-R",
        "RL-F1",
    ]
    rouge_data = [
        [
            m["Model"],
            f"{m['R1-P']:.1f}",
            f"{m['R1-R']:.1f}",
            f"{m['R1-F1']:.1f}",
            f"{m['R2-P']:.1f}",
            f"{m['R2-R']:.1f}",
            f"{m['R2-F1']:.1f}",
            f"{m['RL-P']:.1f}",
            f"{m['RL-R']:.1f}",
            f"{m['RL-F1']:.1f}",
        ]
        for m in all_metrics
    ]
    print(tabulate(rouge_data, headers=rouge_headers, tablefmt="github"))

    # 保存为 CSV
    import csv

    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    # 保存综合表
    with open(
        os.path.join(results_dir, "metrics_summary.csv"),
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(table_data)

    # 保存详细表
    with open(
        os.path.join(results_dir, "metrics_rouge_detailed.csv"),
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(rouge_headers)
        writer.writerows(rouge_data)

    print(f"\n✅ 定量结果已保存至: {results_dir}")


if __name__ == "__main__":
    main()
