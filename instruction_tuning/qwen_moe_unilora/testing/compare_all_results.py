#!/usr/bin/env python
"""
对比 Base Model、Exp1、Exp2、Exp3 和 True Response 的结果
"""
import json
import os

# 路径配置
BASE_DIR = "/root/autodl-tmp/Uni-LoRA/instruction_tuning/qwen_moe_unilora/testing"
PROMPTS_FILE = os.path.join(BASE_DIR, "test_prompts_tcmqa.json")

RESULTS_FILES = {
    "Base Model": os.path.join(BASE_DIR, "test_results_combine/base_only/base_model_results.json"),
    "Exp1 (Joint)": os.path.join(BASE_DIR, "test_results_combine/exp1_joint_only_fixed/trained_model_results.json"),
    "Exp2 (Attn)": os.path.join(BASE_DIR, "test_results_combine/exp2_attn_only/trained_model_results.json"),
    "Exp3 (Full)": os.path.join(BASE_DIR, "test_results_combine/exp3_full_combined/trained_model_results.json"),
}

def load_json(path):
    """加载 JSON 文件"""
    if not os.path.exists(path):
        print(f"警告: 文件不存在: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    # 加载测试提示和真实答案
    prompts_data = load_json(PROMPTS_FILE)
    
    # 创建字典，以 prompt 为键
    data_map = {}
    
    # 从数据集中加载真实答案
    DATASET_FILE = "/root/autodl-tmp/data/fomat/qa_alpaca_dedup_exact.jsonl"
    true_responses = {}
    if os.path.exists(DATASET_FILE):
        with open(DATASET_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    # 支持多种格式
                    prompt = item.get("instruction", "") or item.get("prompt", "") or item.get("input", "")
                    output = item.get("output", "") or item.get("response", "")
                    if prompt:
                        true_responses[prompt.strip()] = output
                except json.JSONDecodeError:
                    pass

    for item in prompts_data:
        p = item.get("prompt", "").strip()
        # 尝试从数据集中查找真实答案
        true_resp = true_responses.get(p, "*数据集中未找到*")
        
        data_map[p] = {
            "Category": item.get("category", ""),
            "True Response": true_resp
        }

    # 加载各个模型的结果
    for model_name, file_path in RESULTS_FILES.items():
        results = load_json(file_path)
        for item in results:
            p = item.get("prompt", "").strip()
            if p in data_map:
                data_map[p][model_name] = item.get("response", "")
            else:
                # 处理提示不匹配的情况
                pass

    # 生成 Markdown 格式的对比报告
    print("# 模型对比报告\n")
    print(f"对比了 {len(data_map)} 个测试案例\n")
    
    for i, (prompt, details) in enumerate(data_map.items(), 1):
        print(f"## 案例 {i}: {prompt}\n")
        print(f"**类别:** {details['Category']}\n")
        
        print(f"### 真实答案")
        print(f"{details['True Response']}\n")
        
        for model_name in RESULTS_FILES.keys():
            response = details.get(model_name, "*(结果未找到)*")
            print(f"### {model_name}")
            print(f"{response}\n")
        
        print("---\n")

if __name__ == "__main__":
    main()
