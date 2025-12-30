"""
使用 FastChat 测试训练前后的模型效果
对比原始模型和训练后的 Uni-LoRA 模型
"""

import argparse
import json
import os
from typing import List

try:
    import torch
    from fastchat.model.model_adapter import (
        get_conversation_template,
        load_model,
    )
except ImportError:
    print("FastChat 未安装，请先安装: pip install fschat")
    print("或者使用 CLI 模式进行手动测试")
    torch = None


def load_model_for_testing(model_path: str, device: str = "cuda"):
    """加载模型用于测试"""
    try:
        model, tokenizer = load_model(
            model_path,
            device=device,
            num_gpus=1,
            max_gpu_memory=None,
            load_8bit=False,
            cpu_offloading=False,
            debug=False,
        )
        return model, tokenizer
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None, None


def generate_response(
    model, tokenizer, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7
) -> str:
    """生成模型响应"""
    try:
        # 获取对话模板
        conv = get_conversation_template("qwen")
        if conv is None:
            conv = get_conversation_template("vicuna")

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()

        # Tokenize
        inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)

        # Generate
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Decode
        output_ids = output_ids[0][len(inputs["input_ids"][0]) :]
        response = tokenizer.decode(output_ids, skip_special_tokens=True)

        # 清理停止词
        stop_str = conv.sep2 if hasattr(conv, "sep2") else conv.sep
        if stop_str:
            response = response.split(stop_str)[0]

        return response.strip()
    except Exception as e:
        print(f"生成响应失败: {e}")
        return ""


def test_with_fastchat_api(
    model_path: str, test_prompts: List[str], output_file: str = None
):
    """使用 FastChat API 测试模型"""
    print(f"\n{'=' * 60}")
    print(f"使用 FastChat API 测试模型: {model_path}")
    print(f"{'=' * 60}\n")

    if torch is None:
        print("PyTorch 未安装，尝试使用 CLI 方式...")
        return test_with_cli_simple(model_path, test_prompts, output_file)

    results = []

    # 加载模型
    print("正在加载模型...")
    try:
        model, tokenizer = load_model(
            model_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
            num_gpus=1,
            max_gpu_memory=None,
            load_8bit=False,
            cpu_offloading=False,
            debug=False,
        )
        print("模型加载成功！\n")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return []

    # 获取对话模板
    conv_template = get_conversation_template("qwen")
    if conv_template is None:
        conv_template = get_conversation_template("vicuna")

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[测试 {i}/{len(test_prompts)}]")
        print(f"提示: {prompt}")
        print("-" * 60)

        try:
            # 构建对话
            conv = conv_template.copy()
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt_text = conv.get_prompt()

            # Tokenize
            inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)

            # Generate
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

            # Decode
            output_ids = output_ids[0][len(inputs["input_ids"][0]) :]
            response = tokenizer.decode(output_ids, skip_special_tokens=True)

            # 清理停止词
            if hasattr(conv, "sep2") and conv.sep2:
                response = response.split(conv.sep2)[0]
            elif hasattr(conv, "sep") and conv.sep:
                response = response.split(conv.sep)[0]

            response = response.strip()
            print(f"响应: {response}")

            results.append(
                {
                    "prompt": prompt,
                    "response": response,
                    "model_path": model_path,
                }
            )
        except Exception as e:
            print(f"生成失败: {e}")
            results.append(
                {
                    "prompt": prompt,
                    "response": f"ERROR: {str(e)}",
                    "model_path": model_path,
                }
            )

    # 保存结果
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {output_file}")

    return results


def test_with_cli_simple(
    model_path: str, test_prompts: List[str], output_file: str = None
):
    """使用简单的 CLI 方式测试（备用方法）"""
    print(f"\n{'=' * 60}")
    print(f"使用 CLI 测试模型: {model_path}")
    print(f"{'=' * 60}\n")

    results = []

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[测试 {i}/{len(test_prompts)}]")
        print(f"提示: {prompt}")
        print("-" * 60)
        print("提示: 请手动运行以下命令进行测试:")
        print(f"  python -m fastchat.serve.cli --model-path {model_path}")
        print("然后在交互界面中输入上述提示。")

        results.append(
            {
                "prompt": prompt,
                "response": "MANUAL_TEST_REQUIRED",
                "model_path": model_path,
            }
        )

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    return results


def compare_models(
    base_model_path: str,
    trained_model_path: str,
    test_prompts: List[str],
    output_dir: str = "./test_results",
):
    """对比两个模型的效果"""
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("模型对比测试")
    print("=" * 80)
    print(f"原始模型: {base_model_path}")
    print(f"训练后模型: {trained_model_path}")
    print(f"测试提示数量: {len(test_prompts)}")
    print("=" * 80)

    # 测试原始模型
    base_results = test_with_fastchat_api(
        base_model_path,
        test_prompts,
        os.path.join(output_dir, "base_model_results.json"),
    )

    # 测试训练后模型
    trained_results = test_with_fastchat_api(
        trained_model_path,
        test_prompts,
        os.path.join(output_dir, "trained_model_results.json"),
    )

    # 生成对比报告
    comparison = []
    for base, trained in zip(base_results, trained_results):
        comparison.append(
            {
                "prompt": base["prompt"],
                "base_model_response": base["response"],
                "trained_model_response": trained["response"],
            }
        )

    comparison_file = os.path.join(output_dir, "comparison.json")
    with open(comparison_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    # 打印对比结果
    print("\n" + "=" * 80)
    print("对比结果")
    print("=" * 80)
    for i, comp in enumerate(comparison, 1):
        print(f"\n[测试 {i}]")
        print(f"提示: {comp['prompt']}")
        print("\n原始模型响应:")
        print(f"  {comp['base_model_response']}")
        print("\n训练后模型响应:")
        print(f"  {comp['trained_model_response']}")
        print("-" * 80)

    print(f"\n详细对比结果已保存到: {comparison_file}")

    return comparison


def main():
    parser = argparse.ArgumentParser(description="使用 FastChat 测试模型效果")
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen1.5-MoE-A2.7B-Chat",
        help="原始模型路径",
    )
    parser.add_argument(
        "--trained-model",
        type=str,
        required=True,
        help="训练后的模型路径（包含 Uni-LoRA 适配器）",
    )
    parser.add_argument(
        "--test-prompts-file",
        type=str,
        default=None,
        help="测试提示文件（JSON 格式，每行一个提示）",
    )
    parser.add_argument(
        "--test-prompts",
        type=str,
        nargs="+",
        default=None,
        help="测试提示列表",
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
        choices=["compare", "single"],
        default="compare",
        help="测试模式: compare (对比) 或 single (仅测试训练后模型)",
    )

    args = parser.parse_args()

    # 准备测试提示
    test_prompts = []

    if args.test_prompts_file:
        with open(args.test_prompts_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                test_prompts = [
                    item.get("prompt", item) if isinstance(item, dict) else item
                    for item in data
                ]
            else:
                test_prompts = [data.get("prompt", str(data))]
    elif args.test_prompts:
        test_prompts = args.test_prompts
    else:
        # 默认测试提示
        test_prompts = [
            "解释一下什么是机器学习。",
            "写一个 Python 函数来计算斐波那契数列。",
            "如何提高工作效率？",
            "介绍一下量子计算的基本原理。",
            "用中文解释一下相对论。",
        ]

    print(f"使用 {len(test_prompts)} 个测试提示")

    # 执行测试
    if args.mode == "compare":
        compare_models(
            args.base_model,
            args.trained_model,
            test_prompts,
            args.output_dir,
        )
    else:
        test_with_fastchat_api(
            args.trained_model,
            test_prompts,
            os.path.join(args.output_dir, "trained_model_results.json"),
        )


if __name__ == "__main__":
    main()
