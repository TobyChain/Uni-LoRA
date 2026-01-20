#!/usr/bin/env python
"""
使用 FastChat 测试训练前后的模型效果
支持原始模型和训练后的 Uni-LoRA 模型对比
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)

# FastChat imports (optional)
try:
    from fastchat.model.model_adapter import (
        get_conversation_template,  # noqa: F401
        load_model as fastchat_load_model,  # noqa: F401
    )

    FASTCHAT_AVAILABLE = True
except ImportError:
    FASTCHAT_AVAILABLE = False
    print("FastChat 未安装，将使用内置加载方法")

# Add modeling path for Uni-LoRA
sys.path.insert(0, str(Path(__file__).parent.parent / "modeling"))
try:
    from modeling_unilora_moe import load_unilora_checkpoint

    UNILORA_AVAILABLE = True
except ImportError:
    UNILORA_AVAILABLE = False
    print("Uni-LoRA 模块未找到")


def load_base_model(
    model_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[Any, Any]:
    """加载基础模型"""
    print(f"加载基础模型: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    return model, tokenizer


# load_unilora_model replaced by shared implementation


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Tuple[str, float]:
    """生成模型响应"""
    messages = [{"role": "user", "content": prompt}]

    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # Fallback for models without chat template
        text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - start_time

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )

    return response.strip(), elapsed


def test_model(
    model,
    tokenizer,
    test_prompts: List[Dict[str, Any]],
    model_name: str,
    output_file: Optional[str] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> List[Dict[str, Any]]:
    """测试单个模型"""
    print(f"\n{'=' * 60}")
    print(f"测试模型: {model_name}")
    print(f"{'=' * 60}\n")

    results = []
    total_time = 0

    for i, item in enumerate(test_prompts, 1):
        prompt = item.get("prompt", item.get("question", str(item)))
        category = item.get("category", "general")

        print(f"\n[测试 {i}/{len(test_prompts)}] ({category})")
        print(f"提示: {prompt}")
        print("-" * 60)

        try:
            response, elapsed = generate_response(
                model,
                tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            total_time += elapsed
            print(
                f"响应 ({elapsed:.2f}s): {response[:200]}{'...' if len(response) > 200 else ''}"
            )

            results.append(
                {
                    "prompt": prompt,
                    "category": category,
                    "response": response,
                    "generation_time": round(elapsed, 2),
                    "model": model_name,
                }
            )
        except Exception as e:
            print(f"生成失败: {e}")
            results.append(
                {
                    "prompt": prompt,
                    "category": category,
                    "response": f"ERROR: {str(e)}",
                    "generation_time": 0,
                    "model": model_name,
                }
            )

    # 统计信息
    print(f"\n{'=' * 60}")
    print(f"测试完成: {len(results)} 个样本, 总耗时 {total_time:.2f}s")
    print(f"平均每样本耗时: {total_time / len(results):.2f}s")

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到: {output_file}")

    return results


def compare_models(
    base_model_path: str,
    trained_model_path: str,
    test_prompts: List[Dict[str, Any]],
    output_dir: str = "./test_results",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> List[Dict[str, Any]]:
    """对比两个模型的效果"""
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("模型对比测试")
    print("=" * 80)
    print(f"原始模型: {base_model_path}")
    print(f"训练后模型: {trained_model_path}")
    print(f"测试提示数量: {len(test_prompts)}")
    print("=" * 80)

    # 检查训练后模型路径
    trained_path = Path(trained_model_path)
    unilora_params_path = trained_path / "unilora_params.pt"

    if not unilora_params_path.exists():
        # Check for adapter_model subdirectory
        adapter_params_path = trained_path / "adapter_model" / "unilora_params.pt"
        if adapter_params_path.exists():
            unilora_params_path = adapter_params_path
            use_unilora = True
        else:
            print("警告: 未找到 unilora_params.pt，将使用 FastChat 加载方式")
            use_unilora = False
    else:
        use_unilora = True

    # 测试原始模型
    print("\n" + "=" * 40)
    print("加载原始模型...")
    base_model, base_tokenizer = load_base_model(base_model_path)
    base_results = test_model(
        base_model,
        base_tokenizer,
        test_prompts,
        "base_model",
        os.path.join(output_dir, "base_model_results.json"),
        max_new_tokens,
        temperature,
    )

    # 释放原始模型内存
    del base_model, base_tokenizer
    torch.cuda.empty_cache()

    # 测试训练后模型
    print("\n" + "=" * 40)
    print("加载训练后模型...")
    if use_unilora:
        trained_model, trained_tokenizer = load_unilora_checkpoint(
            base_model_path,
            str(unilora_params_path),
        )

    else:
        trained_model, trained_tokenizer = load_base_model(trained_model_path)

    trained_results = test_model(
        trained_model,
        trained_tokenizer,
        test_prompts,
        "trained_model",
        os.path.join(output_dir, "trained_model_results.json"),
        max_new_tokens,
        temperature,
    )

    # 生成对比报告
    comparison = []
    for base, trained in zip(base_results, trained_results):
        comparison.append(
            {
                "prompt": base["prompt"],
                "category": base["category"],
                "base_model_response": base["response"],
                "base_model_time": base["generation_time"],
                "trained_model_response": trained["response"],
                "trained_model_time": trained["generation_time"],
            }
        )

    comparison_file = os.path.join(output_dir, "comparison.json")
    with open(comparison_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    # 打印对比结果摘要
    print("\n" + "=" * 80)
    print("对比结果摘要")
    print("=" * 80)
    for i, comp in enumerate(comparison, 1):
        print(f"\n[测试 {i}] {comp['category']}")
        print(f"提示: {comp['prompt']}")
        print(f"\n原始模型 ({comp['base_model_time']:.2f}s):")
        base_resp = comp["base_model_response"]
        print(f"  {base_resp[:150]}{'...' if len(base_resp) > 150 else ''}")
        print(f"\n训练后模型 ({comp['trained_model_time']:.2f}s):")
        trained_resp = comp["trained_model_response"]
        print(f"  {trained_resp[:150]}{'...' if len(trained_resp) > 150 else ''}")
        print("-" * 80)

    print(f"\n详细对比结果已保存到: {comparison_file}")

    return comparison


def load_test_prompts(file_path: str) -> List[Dict[str, Any]]:
    """加载测试提示"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    return [data]


def main():
    parser = argparse.ArgumentParser(description="使用 FastChat 测试模型效果")
    parser.add_argument(
        "--base-model",
        type=str,
        default="/root/autodl-tmp/model/Qwen/Qwen1.5-MoE-A2.7B-Chat",
        help="原始模型路径",
    )
    parser.add_argument(
        "--trained-model",
        type=str,
        default="../training/output/qwen_moe_unilora_pipeline_0107/stage2",
        help="训练后的模型路径（包含 unilora_params.pt）",
    )
    parser.add_argument(
        "--test-prompts-file",
        type=str,
        default="test_prompts.json",
        help="测试提示文件（JSON 格式）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./test_results",
        help="输出目录",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["compare", "single", "base_only"],
        default="compare",
        help="测试模式: compare (对比), single (仅训练后模型), base_only (仅原始模型)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="最大生成 token 数",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="采样温度",
    )

    args = parser.parse_args()

    # 加载测试提示
    if os.path.exists(args.test_prompts_file):
        test_prompts = load_test_prompts(args.test_prompts_file)
    else:
        print(f"警告: 测试提示文件不存在: {args.test_prompts_file}")
        test_prompts = [
            {"prompt": "解释一下什么是机器学习。", "category": "知识问答"},
            {
                "prompt": "写一个 Python 函数来计算斐波那契数列。",
                "category": "代码生成",
            },
            {"prompt": "如何提高工作效率？", "category": "建议咨询"},
        ]

    print(f"使用 {len(test_prompts)} 个测试提示")

    os.makedirs(args.output_dir, exist_ok=True)

    # 执行测试
    if args.mode == "compare":
        compare_models(
            args.base_model,
            args.trained_model,
            test_prompts,
            args.output_dir,
            args.max_new_tokens,
            args.temperature,
        )
    elif args.mode == "single":
        # 仅测试训练后模型
        trained_path = Path(args.trained_model)
        unilora_params_path = trained_path / "unilora_params.pt"

        if not unilora_params_path.exists():
            # Check for adapter_model subdirectory
            adapter_params_path = trained_path / "adapter_model" / "unilora_params.pt"
            if adapter_params_path.exists():
                unilora_params_path = adapter_params_path

        if unilora_params_path.exists():
            model, tokenizer = load_unilora_checkpoint(
                args.base_model, str(unilora_params_path)
            )

        else:
            model, tokenizer = load_base_model(args.trained_model)

        test_model(
            model,
            tokenizer,
            test_prompts,
            "trained_model",
            os.path.join(args.output_dir, "trained_model_results.json"),
            args.max_new_tokens,
            args.temperature,
        )
    else:  # base_only
        model, tokenizer = load_base_model(args.base_model)
        test_model(
            model,
            tokenizer,
            test_prompts,
            "base_model",
            os.path.join(args.output_dir, "base_model_results.json"),
            args.max_new_tokens,
            args.temperature,
        )


if __name__ == "__main__":
    main()
