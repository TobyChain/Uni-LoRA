"""
Uni-LoRA for Qwen MoE Models
Implements parameter-efficient fine-tuning with shared vector bank and expert-specific projections.
"""

import logging
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class UniLoRALinear(nn.Module):
    """
    Uni-LoRA Linear layer that replaces standard Linear layers in MoE experts.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: float,
        shared_vector: nn.Parameter,
        expert_id: Optional[int] = None,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        # Use shared_vector's effective rank for scaling
        effective_rank = 1 if shared_vector.dim() == 1 else shared_vector.size(0)
        self.scaling = alpha / effective_rank

        self.shared_vector = shared_vector

        out_features = base_layer.out_features

        base_device = base_layer.weight.device
        base_dtype = base_layer.weight.dtype

        if str(base_device) == "meta":
            base_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not torch.is_floating_point(torch.empty((), dtype=base_dtype)):
            base_dtype = torch.float16

        if shared_vector.dim() == 1:
            proj_rank = 1
        else:
            # Use shared_vector's first dimension, not lora_rank
            proj_rank = shared_vector.size(0)

        self.projection = nn.Parameter(
            torch.randn(out_features, proj_rank, device=base_device, dtype=base_dtype)
            * 0.01,
            requires_grad=True,
        )

        self.expert_id = expert_id

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base_layer(x)

        if self.shared_vector.dim() == 1:
            vx = torch.matmul(x, self.shared_vector)
            adaptation = self.scaling * (vx.unsqueeze(-1) * self.projection.t())
        else:
            vx = torch.matmul(x, self.shared_vector.t())
            adaptation = self.scaling * torch.matmul(vx, self.projection.t())

        return base_output + adaptation


class UniLoRAMoEExpert(nn.Module):
    """
    Wrapper for a single MoE expert with Uni-LoRA applied to gate_proj, up_proj, and down_proj.
    """

    def __init__(
        self,
        expert: nn.Module,
        rank: int,
        alpha: float,
        shared_vector: nn.Parameter,
        expert_id: int,
    ):
        super().__init__()
        self.expert = expert
        self.expert_id = expert_id

        if hasattr(expert, "gate_proj"):
            self.gate_proj = UniLoRALinear(
                expert.gate_proj, rank, alpha, shared_vector, expert_id
            )
        else:
            self.gate_proj = expert.gate_proj

        if hasattr(expert, "up_proj"):
            self.up_proj = UniLoRALinear(
                expert.up_proj, rank, alpha, shared_vector, expert_id
            )
        else:
            self.up_proj = expert.up_proj

        if hasattr(expert, "down_proj"):
            # Check dimension match for down_proj
            # If dimensions mismatch (e.g. Qwen2-MoE intermediate_size != hidden_size),
            # we cannot use the shared_vector (which has size hidden_size).
            # In this case, we skip Uni-LoRA for down_proj.
            if expert.down_proj.in_features == shared_vector.shape[0]:
                self.down_proj = UniLoRALinear(
                    expert.down_proj, rank, alpha, shared_vector, expert_id
                )
            else:
                self.down_proj = expert.down_proj
        else:
            self.down_proj = expert.down_proj

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        activation = F.silu(gate) * up
        output = self.down_proj(activation)
        return output


class UniLoRAQwen2MoeExperts(nn.Module):
    """
    Wrapper for Qwen2MoeExperts (fused experts) with Uni-LoRA applied.
    """

    def __init__(
        self,
        experts: nn.Module,
        rank: int,
        alpha: float,
        shared_vector: nn.Parameter,
    ):
        super().__init__()
        self.experts = experts
        self.rank = rank
        self.alpha = alpha
        # Use shared_vector's effective rank for scaling
        effective_rank = 1 if shared_vector.dim() == 1 else shared_vector.size(0)
        self.scaling = alpha / effective_rank
        self.shared_vector = shared_vector

        self.num_experts = experts.num_experts
        self.hidden_dim = experts.hidden_dim
        self.intermediate_dim = experts.intermediate_dim
        self.act_fn = experts.act_fn

        base_device = experts.gate_up_proj.device
        base_dtype = experts.gate_up_proj.dtype
        if str(base_device) == "meta":
            base_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not torch.is_floating_point(torch.empty((), dtype=base_dtype)):
            base_dtype = torch.float16

        # Use shared_vector's first dimension when 2D, not lora_rank
        proj_rank = 1 if shared_vector.dim() == 1 else shared_vector.size(0)

        self.gate_up_projections = nn.Parameter(
            torch.randn(
                self.num_experts,
                2 * self.intermediate_dim,
                proj_rank,
                device=base_device,
                dtype=base_dtype,
            )
            * 0.01,
            requires_grad=True,
        )

        if self.intermediate_dim == self.hidden_dim:
            self.down_projections = nn.Parameter(
                torch.randn(
                    self.num_experts,
                    self.hidden_dim,
                    proj_rank,
                    device=base_device,
                    dtype=base_dtype,
                )
                * 0.01,
                requires_grad=True,
            )
        else:
            self.down_projections = None
            logger.warning(
                f"Skipping Uni-LoRA for down_proj: intermediate_dim ({self.intermediate_dim}) != hidden_dim ({self.hidden_dim})"
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)

        if self.shared_vector.dim() == 1:
            vx = torch.matmul(hidden_states, self.shared_vector)
        else:
            vx = torch.matmul(hidden_states, self.shared_vector.t())

        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(
                top_k_index, num_classes=self.num_experts
            )
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx >= self.num_experts:
                continue

            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]

            base_gate_up = nn.functional.linear(
                current_state, self.experts.gate_up_proj[expert_idx]
            )

            current_vx = vx[token_idx]
            if self.shared_vector.dim() == 1:
                adapter_gate_up = self.scaling * (
                    current_vx.unsqueeze(-1) * self.gate_up_projections[expert_idx].t()
                )
            else:
                adapter_gate_up = self.scaling * torch.matmul(
                    current_vx, self.gate_up_projections[expert_idx].t()
                )

            gate_up = base_gate_up + adapter_gate_up
            gate, up = gate_up.chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up

            base_down = nn.functional.linear(
                current_hidden_states, self.experts.down_proj[expert_idx]
            )

            if self.down_projections is not None:
                if self.shared_vector.dim() == 1:
                    vx_down = torch.matmul(current_hidden_states, self.shared_vector)
                    adapter_down = self.scaling * (
                        vx_down.unsqueeze(-1) * self.down_projections[expert_idx].t()
                    )
                else:
                    vx_down = torch.matmul(
                        current_hidden_states, self.shared_vector.t()
                    )
                    adapter_down = self.scaling * torch.matmul(
                        vx_down, self.down_projections[expert_idx].t()
                    )
                current_hidden_states = base_down + adapter_down
            else:
                current_hidden_states = base_down

            current_hidden_states = (
                current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            )
            final_hidden_states.index_add_(
                0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
            )

        return final_hidden_states


class UniLoRAMoELayer(nn.Module):
    """
    Wrapper for Qwen MoE SparseMoeBlock that applies Uni-LoRA to all experts.
    """

    def __init__(
        self,
        moe_block: nn.Module,
        rank: int,
        alpha: float,
        shared_vector: nn.Parameter,
    ):
        super().__init__()
        self.moe_block = moe_block
        self.rank = rank
        self.alpha = alpha
        self.shared_vector = shared_vector

        if hasattr(moe_block, "num_experts"):
            self.num_experts = moe_block.num_experts
        elif hasattr(moe_block, "experts") and hasattr(moe_block.experts, "__len__"):
            self.num_experts = len(moe_block.experts)
        elif hasattr(moe_block, "experts") and hasattr(
            moe_block.experts, "num_experts"
        ):
            self.num_experts = moe_block.experts.num_experts
        else:
            raise ValueError("Cannot determine number of experts from MoE block")

        if hasattr(moe_block, "experts"):
            if isinstance(moe_block.experts, nn.ModuleList):
                self.experts = nn.ModuleList(
                    [
                        UniLoRAMoEExpert(
                            expert=expert,
                            rank=rank,
                            alpha=alpha,
                            shared_vector=shared_vector,
                            expert_id=i,
                        )
                        for i, expert in enumerate(moe_block.experts)
                    ]
                )
            elif moe_block.experts.__class__.__name__ == "Qwen2MoeExperts":
                self.experts = UniLoRAQwen2MoeExperts(
                    experts=moe_block.experts,
                    rank=rank,
                    alpha=alpha,
                    shared_vector=shared_vector,
                )
            else:
                try:
                    self.experts = nn.ModuleList(
                        [
                            UniLoRAMoEExpert(
                                expert=expert,
                                rank=rank,
                                alpha=alpha,
                                shared_vector=shared_vector,
                                expert_id=i,
                            )
                            for i, expert in enumerate(moe_block.experts)
                        ]
                    )
                except TypeError:
                    raise ValueError(
                        f"Unsupported experts type: {type(moe_block.experts)}"
                    )
        else:
            raise ValueError("MoE block does not have 'experts' attribute")

        if hasattr(moe_block, "gate"):
            self.gate = moe_block.gate
        else:
            raise ValueError("MoE block does not have 'gate' attribute")

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        original_experts = self.moe_block.experts
        self.moe_block.experts = self.experts

        try:
            result = self.moe_block(hidden_states, *args, **kwargs)

            # Compute load balancing loss during training
            if self.training and hasattr(self, "gate"):
                with torch.no_grad():
                    batch_size, seq_len, _ = hidden_states.shape
                    hidden_flat = hidden_states.view(-1, hidden_states.size(-1))
                    router_logits = self.gate(hidden_flat)
                    router_probs = F.softmax(router_logits, dim=-1)

                    # Load balancing loss: encourage uniform expert usage
                    # f_i = fraction of tokens routed to expert i
                    # P_i = average routing probability to expert i
                    tokens_per_expert = router_probs.mean(dim=0)  # [num_experts]
                    expert_load = tokens_per_expert * self.num_experts

                    # Variance-based load balancing
                    load_balance_loss = (expert_load.var() + 1e-8).sqrt()
                    self.last_load_balance_loss = load_balance_loss.item()
        finally:
            self.moe_block.experts = original_experts

        return result


def apply_unilora_to_qwen_moe(
    model: nn.Module,
    rank: int = 64,
    alpha: float = 16.0,
    target_modules: Optional[List[str]] = None,
    use_rank1: bool = True,
    shared_vector_rank: int = 8,
) -> nn.Module:
    """
    Apply Uni-LoRA to Qwen MoE model by replacing MoE layers.

    Args:
        model: The base model to apply Uni-LoRA to
        rank: LoRA rank for projections
        alpha: LoRA scaling factor
        target_modules: Optional list of module names to target
        use_rank1: If True, use rank-1 shared vector; if False, use shared_vector_rank
        shared_vector_rank: Rank of shared vector when use_rank1=False
    """
    in_features = None
    for name, module in model.named_modules():
        if hasattr(module, "experts"):
            if isinstance(module.experts, nn.ModuleList) and len(module.experts) > 0:
                first_expert = module.experts[0]
                if hasattr(first_expert, "gate_proj"):
                    in_features = first_expert.gate_proj.in_features
                    break
            elif module.experts.__class__.__name__ == "Qwen2MoeExperts":
                in_features = module.experts.hidden_dim
                break
            elif hasattr(module.experts, "hidden_dim"):
                in_features = module.experts.hidden_dim
                break

    if in_features is None:
        raise ValueError(
            "Could not determine input features dimension from MoE experts"
        )

    if use_rank1:
        shared_vector = nn.Parameter(
            torch.randn(in_features) * 0.01, requires_grad=True
        )
        logger.info(f"Created rank-1 shared vector bank with dimension {in_features}")
    else:
        # Use shared_vector_rank instead of lora rank
        shared_vector = nn.Parameter(
            torch.randn(shared_vector_rank, in_features) * 0.01, requires_grad=True
        )
        logger.info(
            f"Created low-rank shared vector bank with dimension ({shared_vector_rank}, {in_features})"
        )

    ref_device = None
    ref_dtype = None
    try:
        for name, module in model.named_modules():
            if (
                "moe" in name.lower()
                or "sparse_moe" in name.lower()
                or "mlp" in name.lower()
            ) and hasattr(module, "experts"):
                if (
                    isinstance(module.experts, nn.ModuleList)
                    and len(module.experts) > 0
                ):
                    if hasattr(module.experts[0], "gate_proj"):
                        ref_device = module.experts[0].gate_proj.weight.device
                        ref_dtype = module.experts[0].gate_proj.weight.dtype
                        break
                elif module.experts.__class__.__name__ == "Qwen2MoeExperts":
                    ref_device = module.experts.gate_up_proj.device
                    ref_dtype = module.experts.gate_up_proj.dtype
                    break
    except Exception as e:
        logger.warning(f"Failed to infer device/dtype from experts: {e}")

    if ref_device is not None:
        if str(ref_device) == "meta":
            ref_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if ref_dtype is None or not torch.is_floating_point(
            torch.empty((), dtype=ref_dtype)
        ):
            ref_dtype = torch.float16
        shared_vector.data = shared_vector.data.to(device=ref_device, dtype=ref_dtype)
        logger.info(f"Placed shared vector on {ref_device} with dtype {ref_dtype}")

    model.register_parameter("unilora_shared_vector", shared_vector)

    replaced_count = 0
    for name, module in model.named_modules():
        if (
            "moe" in name.lower()
            or "sparse_moe" in name.lower()
            or "mlp" in name.lower()
        ):
            if hasattr(module, "experts") and hasattr(module, "gate"):
                logger.info(f"Replacing MoE block: {name}")

                unilora_moe = UniLoRAMoELayer(
                    moe_block=module,
                    rank=rank,
                    alpha=alpha,
                    shared_vector=shared_vector,
                )

                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]

                if parent_name:
                    parent_module = model
                    for part in parent_name.split("."):
                        parent_module = getattr(parent_module, part)
                    setattr(parent_module, child_name, unilora_moe)
                else:
                    setattr(model, child_name, unilora_moe)

                replaced_count += 1

    logger.info(f"Applied Uni-LoRA to {replaced_count} MoE blocks")

    return model


def freeze_router(model: nn.Module, freeze: bool = True):
    """
    Freeze or unfreeze the router (gate) in MoE layers.
    """
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, UniLoRAMoELayer):
            if hasattr(module, "gate"):
                for param in module.gate.parameters():
                    param.requires_grad = not freeze
                count += 1
                logger.debug(f"{'Frozen' if freeze else 'Unfrozen'} router in {name}")

    logger.info(f"{'Frozen' if freeze else 'Unfrozen'} router in {count} MoE layers")


def freeze_unilora_adapters(model: nn.Module, freeze: bool = True):
    """
    Freeze or unfreeze Uni-LoRA adapter parameters (shared_vector and projections).
    """
    if hasattr(model, "unilora_shared_vector"):
        model.unilora_shared_vector.requires_grad = not freeze
        logger.info(f"{'Frozen' if freeze else 'Unfrozen'} shared vector bank")

    projection_count = 0
    for name, module in model.named_modules():
        if isinstance(module, UniLoRALinear):
            if hasattr(module, "projection"):
                module.projection.requires_grad = not freeze
                projection_count += 1
        elif isinstance(module, UniLoRAQwen2MoeExperts):
            if hasattr(module, "gate_up_projections"):
                module.gate_up_projections.requires_grad = not freeze
                projection_count += 1
            if (
                hasattr(module, "down_projections")
                and module.down_projections is not None
            ):
                module.down_projections.requires_grad = not freeze
                projection_count += 1

    logger.info(
        f"{'Frozen' if freeze else 'Unfrozen'} {projection_count} projection matrices"
    )


def load_unilora_checkpoint(
    base_model_path: str,
    unilora_params_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[nn.Module, any]:
    """
    Load base model and apply Uni-LoRA parameters from checkpoint.
    Returns: (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading base model from {base_model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map="cuda",
        trust_remote_code=True,
    )

    # Load params first to infer configuration
    logger.info(f"Loading Uni-LoRA parameters from {unilora_params_path}")
    unilora_params = torch.load(unilora_params_path, map_location="cpu")

    # Infer use_rank1 from saved shared_vector shape
    use_rank1 = True
    shared_vector_rank = 1
    if "unilora_shared_vector" in unilora_params:
        sv = unilora_params["unilora_shared_vector"]
        if sv.dim() == 2:
            use_rank1 = False
            shared_vector_rank = sv.size(0)
            logger.info(f"Detected shared_vector_rank={shared_vector_rank} (2D)")
        else:
            logger.info("Detected rank-1 shared vector (1D)")

    logger.info(f"Applying Uni-LoRA architecture (use_rank1={use_rank1})...")
    model = apply_unilora_to_qwen_moe(
        model,
        rank=64,  # Default rank
        alpha=16.0,  # Default alpha
        use_rank1=use_rank1,
        shared_vector_rank=shared_vector_rank,
    )

    # Now load parameters
    if "unilora_shared_vector" in unilora_params:
        model.unilora_shared_vector.data = unilora_params["unilora_shared_vector"].to(
            model.unilora_shared_vector.device
        )

    loaded_count = 0
    for name, param in model.named_parameters():
        if name in unilora_params:
            param.data = unilora_params[name].to(param.device)
            loaded_count += 1

    logger.info(f"Loaded {loaded_count} Uni-LoRA projection parameters")
    model.eval()

    return model, tokenizer


# ============================================================
# Attention Layer Uni-LoRA Support (Q/K/V)
# ============================================================


class UniLoRAAttentionLinear(nn.Module):
    """
    Uni-LoRA adapter for attention projection layers (Q, K, V, O).
    Uses a shared vector bank across all attention layers.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: float,
        shared_vector: nn.Parameter,
        layer_name: str = "",
    ):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.shared_vector = shared_vector
        self.layer_name = layer_name

        out_features = base_layer.out_features
        base_device = base_layer.weight.device
        base_dtype = base_layer.weight.dtype

        if str(base_device) == "meta":
            base_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not torch.is_floating_point(torch.empty((), dtype=base_dtype)):
            base_dtype = torch.float16

        proj_rank = 1 if shared_vector.dim() == 1 else shared_vector.size(0)

        self.projection = nn.Parameter(
            torch.randn(out_features, proj_rank, device=base_device, dtype=base_dtype)
            * 0.01,
            requires_grad=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base_layer(x)

        if self.shared_vector.dim() == 1:
            vx = torch.matmul(x, self.shared_vector)
            adaptation = self.scaling * (vx.unsqueeze(-1) * self.projection.t())
        else:
            vx = torch.matmul(x, self.shared_vector.t())
            adaptation = self.scaling * torch.matmul(vx, self.projection.t())

        return base_output + adaptation


def apply_unilora_to_attention(
    model: nn.Module,
    rank: int = 64,
    alpha: float = 16.0,
    target_modules: Optional[List[str]] = None,
    shared_vector_rank: int = 8,
) -> nn.Module:
    """
    Apply Uni-LoRA to attention layers (Q, K, V projections) in the model.

    Args:
        model: The model to apply Uni-LoRA attention adapters to
        rank: LoRA rank for projections
        alpha: LoRA scaling factor
        target_modules: List of attention module names to target (default: q_proj, k_proj, v_proj)
        shared_vector_rank: Rank of shared vector for attention adapters
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj"]

    # Determine hidden dimension from first attention layer
    hidden_dim = None
    for name, module in model.named_modules():
        if hasattr(module, "q_proj") and isinstance(module.q_proj, nn.Linear):
            hidden_dim = module.q_proj.in_features
            break

    if hidden_dim is None:
        raise ValueError("Could not determine hidden dimension from attention layers")

    # Create shared vector for attention layers (separate from MoE shared vector)
    attn_shared_vector = nn.Parameter(
        torch.randn(shared_vector_rank, hidden_dim) * 0.01, requires_grad=True
    )
    logger.info(
        f"Created attention shared vector with dimension ({shared_vector_rank}, {hidden_dim})"
    )

    # Get device and dtype from model
    ref_device = None
    ref_dtype = None
    for param in model.parameters():
        if param.device.type != "meta":
            ref_device = param.device
            ref_dtype = param.dtype
            break

    if ref_device is not None:
        attn_shared_vector.data = attn_shared_vector.data.to(
            device=ref_device, dtype=ref_dtype
        )
        logger.info(f"Placed attention shared vector on {ref_device}")

    model.register_parameter("unilora_attn_shared_vector", attn_shared_vector)

    # Replace attention projection layers
    replaced_count = 0
    for name, module in model.named_modules():
        for target in target_modules:
            if hasattr(module, target):
                original_layer = getattr(module, target)
                if isinstance(original_layer, nn.Linear):
                    new_layer = UniLoRAAttentionLinear(
                        base_layer=original_layer,
                        rank=rank,
                        alpha=alpha,
                        shared_vector=attn_shared_vector,
                        layer_name=f"{name}.{target}",
                    )
                    setattr(module, target, new_layer)
                    replaced_count += 1
                    logger.debug(
                        f"Replaced {name}.{target} with UniLoRAAttentionLinear"
                    )

    logger.info(f"Applied Uni-LoRA to {replaced_count} attention projection layers")
    return model


def freeze_attention_adapters(model: nn.Module, freeze: bool = True):
    """
    Freeze or unfreeze Uni-LoRA attention adapter parameters.
    """
    if hasattr(model, "unilora_attn_shared_vector"):
        model.unilora_attn_shared_vector.requires_grad = not freeze
        logger.info(f"{'Frozen' if freeze else 'Unfrozen'} attention shared vector")

    projection_count = 0
    for name, module in model.named_modules():
        if isinstance(module, UniLoRAAttentionLinear):
            if hasattr(module, "projection"):
                module.projection.requires_grad = not freeze
                projection_count += 1

    logger.info(
        f"{'Frozen' if freeze else 'Unfrozen'} {projection_count} attention projection matrices"
    )
