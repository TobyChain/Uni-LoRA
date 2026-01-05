import torch
from transformers import AutoModelForCausalLM, AutoConfig

model_name = "Qwen/Qwen1.5-MoE-A2.7B-Chat"

print(f"Loading config for {model_name}...")
try:
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    print("Config loaded.")
except Exception as e:
    print(f"Error loading config: {e}")

print(f"Loading model {model_name} (on CPU for inspection)...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="cpu", trust_remote_code=True, torch_dtype=torch.float16
    )
    print("Model loaded.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

print("\nInspecting model structure for MoE layers...")
found_moe = False
for name, module in model.named_modules():
    if "moe" in name.lower():
        print(f"Found module with 'moe' in name: {name}")
        print(f"  Type: {type(module)}")
        if hasattr(module, "experts"):
            print("  Has 'experts' attribute.")
            experts = module.experts
            print(f"  Experts type: {type(experts)}")
            if len(experts) > 0:
                first_expert = experts[0]
                print(f"  First expert type: {type(first_expert)}")
                print(f"  First expert attributes: {dir(first_expert)}")
                if hasattr(first_expert, "gate_proj"):
                    print(
                        f"  First expert has 'gate_proj'. in_features: {first_expert.gate_proj.in_features}"
                    )
                    found_moe = True
                elif hasattr(first_expert, "w1"):
                    print(
                        f"  First expert has 'w1'. in_features: {first_expert.w1.in_features}"
                    )
                elif hasattr(first_expert, "up_proj"):
                    print(
                        f"  First expert has 'up_proj'. in_features: {first_expert.up_proj.in_features}"
                    )
            else:
                print("  Experts list is empty.")
        else:
            print("  Does NOT have 'experts' attribute.")
            # Check if it has other attributes that might hold experts
            print(f"  Attributes: {dir(module)}")

if not found_moe:
    print("\nCould not find the expected MoE structure.")
else:
    print("\nSuccessfully found MoE structure.")
