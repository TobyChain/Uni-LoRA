"""
Uni-LoRA for Qwen MoE Models
Implements parameter-efficient fine-tuning with shared vector bank and expert-specific projections.

Based on qlora_unilora.py implementation, adapted for Qwen MoE architecture.
Supports two-stage training:
  Stage 1: Train experts (freeze router)
  Stage 2: Train router (freeze adapters)
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

    Mathematical formulation:
        ΔW_e = s · (P_e · v^T)
    where:
        - v: Shared vector bank (rank-1 or low rank), shared across ALL experts
        - P_e: Projection matrix, unique to each expert e
        - s: Scaling factor (alpha/r)

    This design follows the Uni-LoRA paper where all experts share the same
    vector bank but have unique projection matrices for task-specific adaptation.
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
        self.scaling = alpha / rank

        # Shared vector bank (v) - reference to the global parameter
        # This is NOT a copy, but a direct reference to ensure gradient flow
        self.shared_vector = shared_vector

        # Expert-specific projection matrix (P_e)
        out_features = base_layer.out_features

        # Determine device and dtype from base layer
        base_device = base_layer.weight.device
        base_dtype = base_layer.weight.dtype

        # If base weights are quantized (non-floating), use fp16 for adapters
        if not torch.is_floating_point(torch.empty((), dtype=base_dtype)):
            base_dtype = torch.float16

        # Determine projection rank based on shared_vector shape
        if shared_vector.dim() == 1:
            # Rank-1 case: P_e is (out_features, 1)
            proj_rank = 1
        else:
            # Low-rank case: P_e is (out_features, rank)
            proj_rank = rank

        # Initialize P_e with small random values (following qlora_unilora.py)
        self.projection = nn.Parameter(
            torch.randn(out_features, proj_rank, device=base_device, dtype=base_dtype)
            * 0.01,
            requires_grad=True,
        )

        self.expert_id = expert_id

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: base_layer(x) + scaling * (P_e @ v^T) @ x

        Efficient computation:
            ΔW_e @ x = s · P_e · (v^T @ x)
        """
        # Base layer output
        base_output = self.base_layer(x)

        # Uni-LoRA adaptation
        if self.shared_vector.dim() == 1:
            # Rank-1 case: v is a vector (in_features,)
            # v^T @ x: (batch, seq_len, in_features) @ (in_features,) -> (batch, seq_len)
            vx = torch.matmul(x, self.shared_vector)
            # P_e is (out_features, 1), P_e.t() is (1, out_features)
            # Broadcasting: (batch, seq_len, 1) * (1, out_features) -> (batch, seq_len, out_features)
            adaptation = self.scaling * (vx.unsqueeze(-1) * self.projection.t())
        else:
            # Low-rank case: v is a matrix (rank, in_features)
            # v^T @ x: (batch, seq_len, in_features) @ (in_features, rank) -> (batch, seq_len, rank)
            vx = torch.matmul(x, self.shared_vector.t())
            # P_e.t() @ vx: (batch, seq_len, rank) @ (rank, out_features) -> (batch, seq_len, out_features)
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

        # Replace linear layers with UniLoRALinear
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
            self.down_proj = UniLoRALinear(
                expert.down_proj, rank, alpha, shared_vector, expert_id
            )
        else:
            self.down_proj = expert.down_proj

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass through the expert with Uni-LoRA adaptations."""
        # Standard MoE expert forward: down_proj(activation(gate_proj(x), up_proj(x)))
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        activation = F.silu(gate) * up
        output = self.down_proj(activation)
        return output


class UniLoRAQwen2MoeExperts(nn.Module):
    """
    Wrapper for Qwen2MoeExperts (fused experts) with Uni-LoRA applied.
    This handles the optimized fused expert implementation in Qwen models.
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
        self.scaling = alpha / rank
        self.shared_vector = shared_vector

        self.num_experts = experts.num_experts
        self.hidden_dim = experts.hidden_dim
        self.intermediate_dim = experts.intermediate_dim
        self.act_fn = experts.act_fn

        # Determine device and dtype
        base_device = experts.gate_up_proj.device
        base_dtype = experts.gate_up_proj.dtype
        if not torch.is_floating_point(torch.empty((), dtype=base_dtype)):
            base_dtype = torch.float16

        proj_rank = 1 if shared_vector.dim() == 1 else rank

        # Create adapter parameters for gate_up_proj
        # gate_up_proj is (num_experts, 2*intermediate, hidden)
        # We need P_e: (2*intermediate, rank) for each expert
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

        # For down_proj, we only add adapters if dimensions match
        # (to reuse the same shared vector)
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
        """
        Forward pass for fused experts with Uni-LoRA adaptations.
        """
        final_hidden_states = torch.zeros_like(hidden_states)

        # Compute shared vector projection once for efficiency
        if self.shared_vector.dim() == 1:
            vx = torch.matmul(hidden_states, self.shared_vector)  # (batch, seq_len)
        else:
            vx = torch.matmul(
                hidden_states, self.shared_vector.t()
            )  # (batch, seq_len, rank)

        # Determine which experts are used
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

            # Base gate_up
            base_gate_up = nn.functional.linear(
                current_state, self.experts.gate_up_proj[expert_idx]
            )

            # Adapter gate_up
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

            # Down proj
            base_down = nn.functional.linear(
                current_hidden_states, self.experts.down_proj[expert_idx]
            )

            if self.down_projections is not None:
                # Compute adapter for down_proj
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

            # Apply routing weights
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
    Maintains the original routing logic while injecting adapter weights.
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

        # Determine number of experts
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

        # Replace experts with Uni-LoRA wrapped experts
        if hasattr(moe_block, "experts"):
            if isinstance(moe_block.experts, nn.ModuleList):
                # Standard ModuleList of experts
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
                # Fused experts implementation
                self.experts = UniLoRAQwen2MoeExperts(
                    experts=moe_block.experts,
                    rank=rank,
                    alpha=alpha,
                    shared_vector=shared_vector,
                )
            else:
                # Fallback for other types
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

        # Keep the gate (router) - will be frozen/unfrozen during training
        if hasattr(moe_block, "gate"):
            self.gate = moe_block.gate
        else:
            raise ValueError("MoE block does not have 'gate' attribute")

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        """
        Forward pass that maintains original MoE routing logic.
        Temporarily replaces experts in moe_block with Uni-LoRA experts.
        """
        # Temporarily replace experts in the original moe_block
        original_experts = self.moe_block.experts
        self.moe_block.experts = self.experts

        # Call the original forward method to maintain routing logic
        try:
            result = self.moe_block(hidden_states, *args, **kwargs)
        finally:
            # Restore original experts
            self.moe_block.experts = original_experts

        return result


def apply_unilora_to_qwen_moe(
    model: nn.Module,
    rank: int = 64,
    alpha: float = 16.0,
    target_modules: Optional[List[str]] = None,
    use_rank1: bool = True,
) -> nn.Module:
    """
    Apply Uni-LoRA to Qwen MoE model by replacing MoE layers.

    This function follows the design from qlora_unilora.py, creating a shared
    vector bank and expert-specific projection matrices.

    Args:
        model: The Qwen MoE model
        rank: Rank of the low-rank adaptation (r)
        alpha: Scaling factor (alpha)
        target_modules: List of module names to target (default: all MoE blocks)
        use_rank1: Whether to use rank-1 shared vector (default: True for memory efficiency)

    Returns:
        Modified model with Uni-LoRA applied
    """
    # Find a reference layer to get dimensions
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

    # Create global shared vector bank
    # Following qlora_unilora.py: rank-1 for simplicity, or low-rank for more capacity
    if use_rank1:
        # Rank-1 mode: single vector shared across all experts
        shared_vector = nn.Parameter(
            torch.randn(in_features) * 0.01, requires_grad=True
        )
        logger.info(f"Created rank-1 shared vector bank with dimension {in_features}")
    else:
        # Low-rank mode: matrix with rank rows
        shared_vector = nn.Parameter(
            torch.randn(rank, in_features) * 0.01, requires_grad=True
        )
        logger.info(
            f"Created low-rank shared vector bank with dimension ({rank}, {in_features})"
        )

    # Place shared vector on the same device/dtype as a reference expert weight
    ref_device = None
    ref_dtype = None
    try:
        for name, module in model.named_modules():
            if ("moe" in name.lower() or "sparse_moe" in name.lower()) and hasattr(
                module, "experts"
            ):
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
        if ref_dtype is None or not torch.is_floating_point(
            torch.empty((), dtype=ref_dtype)
        ):
            ref_dtype = torch.float16
        shared_vector.data = shared_vector.data.to(device=ref_device, dtype=ref_dtype)
        logger.info(f"Placed shared vector on {ref_device} with dtype {ref_dtype}")

    # Register shared vector as a model-level parameter
    model.register_parameter("unilora_shared_vector", shared_vector)

    # Find and replace MoE blocks
    replaced_count = 0
    for name, module in model.named_modules():
        # Check if this is a MoE block
        if "moe" in name.lower() or "sparse_moe" in name.lower():
            if hasattr(module, "experts") and hasattr(module, "gate"):
                logger.info(f"Replacing MoE block: {name}")

                # Create UniLoRAMoELayer
                unilora_moe = UniLoRAMoELayer(
                    moe_block=module,
                    rank=rank,
                    alpha=alpha,
                    shared_vector=shared_vector,
                )

                # Replace the module in the parent
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

    Args:
        model: The model with Uni-LoRA MoE layers
        freeze: If True, freeze the router; if False, unfreeze it
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

    Args:
        model: The model with Uni-LoRA MoE layers
        freeze: If True, freeze adapters; if False, unfreeze them
    """
    # Freeze shared vector
    if hasattr(model, "unilora_shared_vector"):
        model.unilora_shared_vector.requires_grad = not freeze
        logger.info(f"{'Frozen' if freeze else 'Unfrozen'} shared vector bank")

    # Freeze expert-specific projections
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
