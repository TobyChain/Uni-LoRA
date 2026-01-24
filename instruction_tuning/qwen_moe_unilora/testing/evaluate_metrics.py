
import json
import os
import jieba
from rouge_score import rouge_scorer
import sacrebleu
import numpy as np

# 路径配置
BASE_DIR = "/root/autodl-tmp/Uni-LoRA/instruction_tuning/qwen_moe_unilora/testing"
DATASET_FILE = "/root/autodl-tmp/data/fomat/qa_alpaca_dedup_exact.jsonl"
PROMPTS_FILE = os.path.join(BASE_DIR, "test_prompts_tcmqa.json")

RESULTS_FILES = {
    "Base Model": os.path.join(BASE_DIR, "test_results_combine/base_only/base_model_results.json"),
    "Exp1 (Joint)": os.path.join(BASE_DIR, "test_results_combine/exp1_joint_only_fixed/trained_model_results.json"),
    "Exp2 (Attn)": os.path.join(BASE_DIR, "test_results_combine/exp2_attn_only/trained_model_results.json"),
    "Exp3 (Full)": os.path.join(BASE_DIR, "test_results_combine/exp3_full_combined/trained_model_results.json"),
}

def load_json(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def tokenize_chinese(text):
    """使用 jieba 进行分词"""
    return " ".join(jieba.cut(text))

def calculate_metrics(hyps, refs):
    """计算 BLEU 和 ROUGE 分数"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    
    bleu_scores = []
    r1_scores = []
    r2_scores = []
    rl_scores = []
    
    for hyp, ref in zip(hyps, refs):
        # 分词
        hyp_tok = tokenize_chinese(hyp)
        ref_tok = tokenize_chinese(ref)
        
        # BLEU (sacrebleu 默认处理空格分隔的词)
        bleu = sacrebleu.sentence_bleu(hyp_tok, [ref_tok]).score
        bleu_scores.append(bleu)
        
        # ROUGE
        scores = scorer.score(ref_tok, hyp_tok)
        r1_scores.append(scores['rouge1'].fmeasure * 100)
        r2_scores.append(scores['rouge2'].fmeasure * 100)
        rl_scores.append(scores['rougeL'].fmeasure * 100)
        
    return {
        "BLEU": np.mean(bleu_scores),
        "ROUGE-1": np.mean(r1_scores),
        "ROUGE-2": np.mean(r2_scores),
        "ROUGE-L": np.mean(rl_scores),
    }

def main():
    # 1. 加载提示词
    prompts_data = load_json(PROMPTS_FILE)
    prompts_list = [item.get("prompt", "").strip() for item in prompts_data]
    
    # 2. 加载数据集中的真实答案 (Ground Truth)
    true_responses = {}
    if os.path.exists(DATASET_FILE):
        with open(DATASET_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    p = (item.get("instruction", "") or item.get("prompt", "") or item.get("input", "")).strip()
                    o = item.get("output", "") or item.get("response", "")
                    if p:
                        true_responses[p] = o
                except:
                    pass

    # 3. 计算每个模型的指标
    all_metrics = {}
    
    for model_name, file_path in RESULTS_FILES.items():
        print(f"正在计算 {model_name} 的指标...")
        results = load_json(file_path)
        
        hyps = []
        refs = []
        
        # 创建结果映射以便对齐提示词
        results_map = {item.get("prompt", "").strip(): item.get("response", "") for item in results}
        
        for p in prompts_list:
            if p in results_map and p in true_responses:
                hyps.append(results_map[p])
                refs.append(true_responses[p])
        
        if not hyps:
            print(f"警告: {model_name} 没有匹配到有效的结果。")
            continue
            
        metrics = calculate_metrics(hyps, refs)
        all_metrics[model_name] = metrics

    # 4. 打印汇总结果
    print("\n" + "="*80)
    print(f"{'模型名称':<15} | {'BLEU':<8} | {'ROUGE-1':<10} | {'ROUGE-2':<10} | {'ROUGE-L':<10}")
    print("-" * 80)
    
    for model_name, m in all_metrics.items():
        print(f"{model_name:<15} | {m['BLEU']:<8.2f} | {m['ROUGE-1']:<10.2f} | {m['ROUGE-2']:<10.2f} | {m['ROUGE-L']:<10.2f}")
    print("="*80)

if __name__ == "__main__":
    main()
