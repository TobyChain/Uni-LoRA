"""
Uni-LoRA for Qwen MoE Models
Implements parameter-efficient fine-tuning with shared vector bank and expert-specific projections.
"""

import logging
from typing import Optional

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

        # Shared vector bank (v) - shared across all experts
        self.register_buffer("shared_vector", shared_vector)

        # Expert-specific projection matrix (P_e)
        out_features = base_layer.out_features

        # Determine the actual rank for projection
        # If shared_vector is rank-1 (dim=1), then P_e should be (out_features, 1)
        # If shared_vector is low-rank (dim=2), then P_e should be (out_features, rank)
        if shared_vector.dim() == 1:
            # Rank-1 case: P_e is (out_features, 1)
            proj_rank = 1
        else:
            # Low-rank case: P_e is (out_features, rank)
            proj_rank = rank

        # Initialize P_e with small random values
        self.projection = nn.Parameter(
            torch.randn(out_features, proj_rank) * 0.01, requires_grad=True
        )

        self.expert_id = expert_id

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: base_layer(x) + scaling * (P_e @ v^T) @ x

        Mathematical formulation:
            ΔW_e = s · (P_e · v^T)
            where v: (r, in_features) or (in_features,) for rank-1
                  P_e: (out_features, r) or (out_features, 1) for rank-1
                  s = alpha / r

        Efficient computation:
            ΔW_e @ x = s · P_e · (v^T @ x)
        """
        # Base layer output
        base_output = self.base_layer(x)

        # Uni-LoRA adaptation: ΔW_e @ x = s · (P_e · v^T) @ x
        # = s · P_e @ (v^T @ x)
        # More efficient: compute v^T @ x first, then P_e @ result
        if self.shared_vector.dim() == 1:
            # Rank-1 case: v is a vector (in_features,)
            # v^T @ x: (batch, seq_len, in_features) @ (in_features,) -> (batch, seq_len)
            vx = torch.matmul(x, self.shared_vector)  # (batch, seq_len)
            # P_e is (out_features, 1), so we need to expand vx and multiply
            vx_expanded = vx.unsqueeze(-1)  # (batch, seq_len, 1)
            # P_e.t() @ vx: (1, out_features) @ (batch, seq_len, 1) -> (batch, seq_len, out_features)
            adaptation = self.scaling * torch.matmul(
                vx_expanded, self.projection.t()
            )  # (batch, seq_len, out_features)
        else:
            # Low-rank case: v is a matrix (rank, in_features)
            # v^T @ x: (rank, in_features) @ (batch, seq_len, in_features) -> (batch, seq_len, rank)
            vx = torch.matmul(x, self.shared_vector.t())  # (batch, seq_len, rank)
            # P_e.t() @ vx: (rank, out_features) @ (batch, seq_len, rank) -> (batch, seq_len, out_features)
            adaptation = self.scaling * torch.matmul(
                vx, self.projection.t()
            )  # (batch, seq_len, out_features)

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

        # Replace gate_proj, up_proj, down_proj with UniLoRALinear
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

        # Store reference to shared vector
        self.register_buffer("shared_vector", shared_vector)

        # Get the number of experts
        if hasattr(moe_block, "num_experts"):
            self.num_experts = moe_block.num_experts
        elif hasattr(moe_block, "experts") and hasattr(moe_block.experts, "__len__"):
            self.num_experts = len(moe_block.experts)
        else:
            raise ValueError("Cannot determine number of experts from MoE block")

        # Replace each expert with UniLoRAMoEExpert
        if hasattr(moe_block, "experts"):
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
        else:
            raise ValueError("MoE block does not have 'experts' attribute")

        # Keep the gate (router) as is - it will be frozen/unfrozen during training
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
    target_modules: Optional[list] = None,
) -> nn.Module:
    """
    Apply Uni-LoRA to Qwen MoE model by replacing MoE layers.

    Args:
        model: The Qwen MoE model
        rank: Rank of the low-rank adaptation (r)
        alpha: Scaling factor (alpha)
        target_modules: List of module names to target (default: all MoE blocks)

    Returns:
        Modified model with Uni-LoRA applied
    """
    # Determine the dimension for shared vector
    # For rank-1: shared_vector is (in_features,)
    # For low-rank: shared_vector is (rank, in_features)
    # We'll use rank-1 for simplicity and memory efficiency
    use_rank1 = True

    # Find a reference layer to get dimensions
    in_features = None
    for name, module in model.named_modules():
        if "moe" in name.lower() and hasattr(module, "experts"):
            if len(module.experts) > 0:
                first_expert = module.experts[0]
                if hasattr(first_expert, "gate_proj"):
                    in_features = first_expert.gate_proj.in_features
                    break

    if in_features is None:
        raise ValueError(
            "Could not determine input features dimension from MoE experts"
        )

    # Create global shared vector bank
    if use_rank1:
        # Rank-1: single vector shared across all experts
        shared_vector = nn.Parameter(
            torch.randn(in_features) * 0.01, requires_grad=True
        )
    else:
        # Low-rank: matrix shared across all experts
        shared_vector = nn.Parameter(
            torch.randn(rank, in_features) * 0.01, requires_grad=True
        )

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

    logger.info(f"Replaced {replaced_count} MoE blocks with Uni-LoRA")

    return model


def freeze_router(model: nn.Module, freeze: bool = True):
    """
    Freeze or unfreeze the router (gate) in MoE layers.

    Args:
        model: The model with Uni-LoRA MoE layers
        freeze: If True, freeze the router; if False, unfreeze it
    """
    for name, module in model.named_modules():
        if isinstance(module, UniLoRAMoELayer):
            if hasattr(module, "gate"):
                for param in module.gate.parameters():
                    param.requires_grad = not freeze
                logger.info(f"{'Frozen' if freeze else 'Unfrozen'} router in {name}")


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

    # Freeze expert-specific projections
    for name, module in model.named_modules():
        if isinstance(module, UniLoRALinear):
            if hasattr(module, "projection"):
                module.projection.requires_grad = not freeze
            logger.debug(
                f"{'Frozen' if freeze else 'Unfrozen'} Uni-LoRA adapter in {name}"
            )
