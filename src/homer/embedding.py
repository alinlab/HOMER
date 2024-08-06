"""
Modified code from the following sources:
 - https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
 - https://github.com/jquesnelle/yarn/blob/master/scaled_rope/patch.py
"""

import math

import torch
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding


# Inverse dim formula to find dim based on number of rotations
def find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


# Find dim range bounds based on rotations
def find_correction_range(
    low_rot, high_rot, dim, base=10000, max_position_embeddings=2048
):
    low = math.floor(find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


def get_mscale(scale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0


class LlamaYaRNScaledRotaryEmbedding(LlamaRotaryEmbedding):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        scaling_factor=1.0,
        original_max_position_embeddings=2048,
        device=None,
    ):
        super().__init__(
            dim=dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
            device=device,
            scaling_factor=scaling_factor,
        )

        extrapolation_factor = 1
        attn_factor = 1
        beta_fast = 32
        beta_slow = 1

        self.original_max_position_embeddings = original_max_position_embeddings
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

        self.yarn(device)

        self.max_seq_len_cached = max_position_embeddings

    def forward(self, x, position_ids):
        cos, sin = super().forward(x, position_ids)
        return cos * self.mscale, sin * self.mscale

    def yarn(self, device):
        pos_freqs = self.base ** (
            torch.arange(0, self.dim, 2).float().to(device) / self.dim
        )
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (self.scaling_factor * pos_freqs)

        low, high = find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.dim,
            self.base,
            self.original_max_position_embeddings,
        )
        inv_freq_mask = (
            1 - linear_ramp_mask(low, high, self.dim // 2).float().to(device)
        ) * self.extrapolation_factor  # Get n-d rotational scaling corrected for extrapolation
        inv_freq = (
            inv_freq_interpolation * (1 - inv_freq_mask)
            + inv_freq_extrapolation * inv_freq_mask
        )

        self.register_buffer("inv_freq", inv_freq)
        self.mscale = float(
            get_mscale(self.scaling_factor) * self.attn_factor
        )  # Get n-d magnitude scaling corrected for interpolation


def patch_llama_for_yarn(
    model, scale, original_max_position_embeddings, max_position_embeddings
):
    for each in model.model.layers:
        each.self_attn.rotary_emb = LlamaYaRNScaledRotaryEmbedding(
            each.self_attn.head_dim,
            scaling_factor=scale,
            original_max_position_embeddings=original_max_position_embeddings,
            max_position_embeddings=max_position_embeddings,  # Set to arbitrary large value
            device=each.self_attn.rotary_emb.inv_freq.device,
        )
