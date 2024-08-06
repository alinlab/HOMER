import torch
import math

from transformers.cache_utils import DynamicCache


class Chunk:
    def __init__(
        self,
        # Uppermost layer info
        hidden_states,
        position_ids,
        attention_mask,
        reduction_mask,
        prefix_len=0,
        suffix_len=0,
        # Visualization
        visualize=False,
    ):
        self.hidden_states = hidden_states
        self.position_ids = position_ids
        self.attention_mask = attention_mask
        self.reduction_mask = reduction_mask

        self.prefix_len = prefix_len
        self.suffix_len = suffix_len

        assert prefix_len > 0 or suffix_len > 0, "Affix length must be greater than 0"

        # Visualization variables (curr, left, right)
        self.visualize = visualize
        self.prune_masks = (None, None, None)
        self.sig_weights = (None, None, None)

        # Cache
        self.cache = DynamicCache()

    @property
    def sequence_len(self):
        # Potential bug? (hidden_states can be input_ids)
        return self.hidden_states.size(1)

    @property
    def prefix_end_at(self):
        return self.prefix_len

    @property
    def context_end_at(self):
        return -self.suffix_len

    # Hidden states
    @property
    def prefix_hidden_states(self):
        end_at = self.prefix_end_at
        return self.hidden_states[:, :end_at, :]

    @property
    def context_hidden_states(self):
        start_at = self.prefix_end_at
        end_at = self.context_end_at

        return self.hidden_states[:, start_at:end_at, :]

    @property
    def suffix_hidden_states(self):
        start_at = self.context_end_at
        return self.hidden_states[:, start_at:, :]

    # KV cache (key)
    @property
    def prefix_key_cache(self):
        end_at = self.prefix_end_at
        return [key_cache[:, :, :end_at, :] for key_cache in self.cache.key_cache]

    @property
    def context_key_cache(self):
        start_at = self.prefix_end_at
        end_at = self.context_end_at

        return [
            key_cache[:, :, start_at:end_at, :] for key_cache in self.cache.key_cache
        ]

    @property
    def suffix_key_cache(self):
        start_at = self.context_end_at
        return [key_cache[:, :, start_at:, :] for key_cache in self.cache.key_cache]

    # KV cache (value)
    @property
    def prefix_value_cache(self):
        end_at = self.prefix_end_at
        return [value_cache[:, :, :end_at, :] for value_cache in self.cache.value_cache]

    @property
    def context_value_cache(self):
        start_at = self.prefix_end_at
        end_at = self.context_end_at

        return [
            value_cache[:, :, start_at:end_at, :]
            for value_cache in self.cache.value_cache
        ]

    @property
    def suffix_value_cache(self):
        start_at = self.context_end_at
        return [
            value_cache[:, :, start_at:, :] for value_cache in self.cache.value_cache
        ]

    # Position IDs
    @property
    def prefix_position_ids(self):
        end_at = self.prefix_end_at
        return self.position_ids[:, :end_at]

    @property
    def context_position_ids(self):
        start_at = self.prefix_end_at
        end_at = self.context_end_at
        return self.position_ids[:, start_at:end_at]

    @property
    def suffix_position_ids(self):
        start_at = self.context_end_at
        return self.position_ids[:, start_at:]

    # Attention mask
    @property
    def prefix_attention_mask(self):
        end_at = self.prefix_end_at
        return self.attention_mask[:, :end_at]

    @property
    def context_attention_mask(self):
        start_at = self.prefix_end_at
        end_at = self.context_end_at
        return self.attention_mask[:, start_at:end_at]

    @property
    def suffix_attention_mask(self):
        start_at = self.context_end_at
        return self.attention_mask[:, start_at:]

    # Reduction mask
    @property
    def prefix_reduction_mask(self):
        end_at = self.prefix_len
        return self.reduction_mask[:, :end_at]

    @property
    def context_reduction_mask(self):
        start_at = self.prefix_end_at
        end_at = self.context_end_at
        return self.reduction_mask[:, start_at:end_at]

    @property
    def suffix_reduction_mask(self):
        start_at = self.context_end_at
        return self.reduction_mask[:, start_at:]

    def update_hidden_states(self, hidden_states):
        self.hidden_states = hidden_states

    def prune(self, prune_mask):
        # Prune position ids, attention mask, reduction mask, cache

        if prune_mask is None:
            return

        self.position_ids = self.position_ids[:, prune_mask]
        self.attention_mask = self.attention_mask[:, prune_mask]
        self.reduction_mask = self.reduction_mask[:, prune_mask]

        self.cache.key_cache = [
            key_cache[:, :, prune_mask, :] for key_cache in self.cache.key_cache
        ]
        self.cache.value_cache = [
            value_cache[:, :, prune_mask, :] for value_cache in self.cache.value_cache
        ]

    def set_visualization_info(self, prune_mask, sig_weight):
        if not self.visualize:
            return

        prune_mask = prune_mask[self.prefix_len : -self.suffix_len]
        sig_weight = sig_weight.squeeze()[self.prefix_len : -self.suffix_len]

        self.prune_masks = (
            prune_mask.detach().cpu(),
            self.prune_masks[1],
            self.prune_masks[2],
        )
        self.sig_weights = (
            sig_weight.detach().cpu(),
            self.sig_weights[1],
            self.sig_weights[2],
        )

    def get_visualization_info(self):
        """
        Output: Prefix length, suffix length, prune masks, significance weights
        """
        return self.prefix_len, self.suffix_len, self.prune_masks, self.sig_weights

    @staticmethod
    def merge(left, right):
        # If right chunk is None, only update visualization info
        if right is None:
            if left.visualize:
                left.prune_masks = (None, left.prune_masks, None)
                left.sig_weights = (None, left.sig_weights, None)

            return left

        # Hidden states
        prefix_hidden_states = (
            left.prefix_hidden_states + right.prefix_hidden_states
        ) / 2
        suffix_hidden_states = (
            left.suffix_hidden_states + right.suffix_hidden_states
        ) / 2
        merged_hidden_states = torch.cat(
            [
                prefix_hidden_states,
                left.context_hidden_states,
                right.context_hidden_states,
                suffix_hidden_states,
            ],
            dim=1,
        )

        left.hidden_states = merged_hidden_states

        # KV cache

        merged_key_cache = []
        merged_value_cache = []

        for (
            left_prefix_key_cache,
            left_context_key_cache,
            left_suffix_key_cache,
            left_prefix_value_cache,
            left_context_value_cache,
            left_suffix_value_cache,
            right_prefix_key_cache,
            right_context_key_cache,
            right_suffix_key_cache,
            right_prefix_value_cache,
            right_context_value_cache,
            right_suffix_value_cache,
        ) in zip(
            left.prefix_key_cache,
            left.context_key_cache,
            left.suffix_key_cache,
            left.prefix_value_cache,
            left.context_value_cache,
            left.suffix_value_cache,
            right.prefix_key_cache,
            right.context_key_cache,
            right.suffix_key_cache,
            right.prefix_value_cache,
            right.context_value_cache,
            right.suffix_value_cache,
        ):
            prefix_key_cache = (left_prefix_key_cache + right_prefix_key_cache) / 2
            suffix_key_cache = (left_suffix_key_cache + right_suffix_key_cache) / 2
            prefix_value_cache = (
                left_prefix_value_cache + right_prefix_value_cache
            ) / 2
            suffix_value_cache = (
                left_suffix_value_cache + right_suffix_value_cache
            ) / 2

            merged_key_cache.append(
                torch.cat(
                    [
                        prefix_key_cache,
                        left_context_key_cache,
                        right_context_key_cache,
                        suffix_key_cache,
                    ],
                    dim=2,
                )
            )

            merged_value_cache.append(
                torch.cat(
                    [
                        prefix_value_cache,
                        left_context_value_cache,
                        right_context_value_cache,
                        suffix_value_cache,
                    ],
                    dim=2,
                )
            )

        left.cache.key_cache = merged_key_cache
        left.cache.value_cache = merged_value_cache

        # Position IDs
        merged_position_ids = torch.cat(
            [
                left.prefix_position_ids,
                left.context_position_ids,
                right.context_position_ids,
                right.suffix_position_ids,
            ],
            dim=1,
        )

        left.position_ids = merged_position_ids

        # Attention mask
        merged_attention_mask = torch.cat(
            [
                left.prefix_attention_mask,
                left.context_attention_mask,
                right.context_attention_mask,
                right.suffix_attention_mask,
            ],
            dim=1,
        )

        left.attention_mask = merged_attention_mask

        # Reduction mask
        merged_reduction_mask = torch.cat(
            [
                left.prefix_reduction_mask,
                left.context_reduction_mask,
                right.context_reduction_mask,
                right.suffix_reduction_mask,
            ],
            dim=1,
        )

        left.reduction_mask = merged_reduction_mask

        if left.visualize and right.visualize:
            left.prune_masks = (None, left.prune_masks, right.prune_masks)
            left.sig_weights = (None, left.sig_weights, right.sig_weights)
        else:
            left.prune_masks = None
            left.sig_weights = None

        del right

        torch.cuda.empty_cache()

        return left

    @staticmethod
    def make_chunks(
        prefix_ids,
        context_ids,
        suffix_ids,
        num_chunks,
        eff_chunk_len,
        visualize=False,
    ):
        assert context_ids is not None, "Context input IDs must be provided"

        device = context_ids.device

        full_context_len = context_ids.size(1)
        prefix_len = prefix_ids.size(1) if prefix_ids is not None else 0
        suffix_len = suffix_ids.size(1) if suffix_ids is not None else 0

        chunks = []
        for i in range(num_chunks):
            start_idx = i * eff_chunk_len

            end_idx = min((i + 1) * eff_chunk_len, full_context_len)
            context_len = end_idx - start_idx  # Contains overlap

            # Input ids
            input_ids = torch.cat(
                [
                    prefix_ids,
                    context_ids[:, start_idx:end_idx],
                    suffix_ids,
                ],
                dim=1,
            )

            # Position ids
            position_ids = torch.arange(prefix_len + context_len + suffix_len)

            if num_chunks > 1:
                # For the last chunk
                suffix_pos_offset = prefix_len + eff_chunk_len
                position_ids[-suffix_len:] = (
                    torch.arange(suffix_len) + suffix_pos_offset
                )

            position_ids = position_ids.unsqueeze(0).to(device)

            # Attention mask
            attention_mask = torch.ones_like(position_ids).to(device)

            # Reduction mask
            reduction_mask = torch.cat(
                [
                    torch.zeros((1, prefix_len), dtype=torch.bool),
                    torch.ones((1, context_len), dtype=torch.bool),
                    torch.zeros((1, suffix_len), dtype=torch.bool),
                ],
                dim=1,
            )

            chunk = Chunk(
                hidden_states=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                reduction_mask=reduction_mask,
                prefix_len=prefix_len,
                suffix_len=suffix_len,
                visualize=visualize,
            )
            chunks.append(chunk)

        # Fill None to match power of 2
        num_full_chunks = 2 ** math.ceil(math.log2(num_chunks))
        chunks = chunks + [None] * (num_full_chunks - num_chunks)

        return chunks
