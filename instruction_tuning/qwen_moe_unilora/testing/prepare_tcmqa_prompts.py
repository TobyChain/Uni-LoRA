import json
import random

input_file = "/root/autodl-tmp/data/fomat/qa_alpaca_clean.jsonl"
output_file = "/root/autodl-tmp/Uni-LoRA/instruction_tuning/qwen_moe_unilora/testing/test_prompts_tcmqa.json"
sample_size = 50

print(f"Reading from {input_file}...")
lines = []
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            lines.append(line)

print(f"Total lines: {len(lines)}")
sampled_lines = random.sample(lines, min(sample_size, len(lines)))

prompts = []
for line in sampled_lines:
    data = json.loads(line)
    prompts.append({"prompt": data["instruction"], "category": "TCMQA"})

print(f"Writing {len(prompts)} prompts to {output_file}...")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(prompts, f, indent=4, ensure_ascii=False)

print("Done.")
