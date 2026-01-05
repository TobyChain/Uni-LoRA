try:
    from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeExperts

    print("Successfully imported Qwen2MoeExperts")
    print(f"Type: {type(Qwen2MoeExperts)}")
    print(f"Dir: {dir(Qwen2MoeExperts)}")

    # Try to instantiate or inspect __init__
    import inspect

    print(f"Init signature: {inspect.signature(Qwen2MoeExperts.__init__)}")
except ImportError:
    print("Could not import Qwen2MoeExperts directly")
    # Try to load config to see if we can get it
    from transformers import AutoConfig

    try:
        config = AutoConfig.from_pretrained(
            "Qwen/Qwen1.5-MoE-A2.7B-Chat", trust_remote_code=True
        )
        print("Config loaded")
    except Exception as e:
        print(f"Error loading config: {e}")
