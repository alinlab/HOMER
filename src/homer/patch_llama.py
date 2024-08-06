import math
import types
from typing import List, Optional, Tuple

import torch
from transformers import (
    Cache,
    LlamaForCausalLM,
    StaticCache,
)
from transformers.models.llama.modeling_llama import (
    LlamaFlashAttention2,
    apply_rotary_pos_emb,
)
from transformers.utils import logging

from .chunk import Chunk

logger = logging.get_logger(__name__)


class MergeManager:
    def __init__(
        self,
        max_chunk_len=2048,
        max_initial_chunk_len=-1,
        reduction_mode="power_max_last_calibrated",
        layers_warmup=12,
        target_len=2048,
        bias_path=None,
        visualize=False,
    ):
        # Values updated per experiment
        self.max_chunk_len = max_chunk_len
        self.max_initial_chunk_len = max_initial_chunk_len
        self.reduction_mode = reduction_mode
        self.layers_warmup = layers_warmup
        self.target_len = target_len
        self.visualize = visualize
        self.bias = (
            torch.load(bias_path, map_location="cpu") if bias_path is not None else None
        )

        # Values updated per sample
        self.prefix_len = None
        self.suffix_len = None
        self.eff_max_chunk_len = None
        self.layers_per_chunk = None
        self.layers_leftover = None

        # Values updated per layer
        self.layer_reduction_info = None
        self.layer_reduction_results = None

    def set_sample_info(
        self,
        prefix_len,
        suffix_len,
        eff_max_chunk_len,
        layers_per_chunk,
        layers_leftover,
    ):
        self.prefix_len = prefix_len
        self.suffix_len = suffix_len
        self.eff_max_chunk_len = eff_max_chunk_len
        self.layers_per_chunk = layers_per_chunk
        self.layers_leftover = layers_leftover

    def set_layer_reduction_info(
        self,
        num_tokens_to_reduce=0,
        reduction_mask=None,
        position_ids=None,
    ):
        if num_tokens_to_reduce == 0:
            self.layer_reduction_info = None
        else:
            self.layer_reduction_info = {
                "num_tokens_to_reduce": num_tokens_to_reduce,
                "reduction_mask": reduction_mask,
                "position_ids": position_ids,
            }

    def set_layer_reduction_results(
        self,
        position_ids=None,
        prune_mask=None,
        significance_weights=None,
    ):
        self.layer_reduction_results = {
            "position_ids": position_ids,
            "prune_mask": prune_mask,
            "significance_weights": significance_weights,
        }


# Class method for LlamaFlashAttention2
def llamaflashattention2_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    Modification: If output_attentions is True, the attention weights corresponding to the final query are computed and returned.
    """
    if isinstance(past_key_value, StaticCache):
        raise ValueError(
            "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
            "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
        )

    #################### MODIFIED FROM ORIGINAL CODE #####################

    # output_attentions = False

    ######################################################################

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(
        bsz, q_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
    # to be able to avoid many of these transpose/reshape/view.
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    dropout_rate = self.attention_dropout if self.training else 0.0

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (LlamaRMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    attn_output = self._flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        q_len,
        dropout=dropout_rate,
    )

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    #################### MODIFIED FROM ORIGINAL CODE #####################

    if output_attentions:
        tmp_query_states = query_states[:, -1:, :, :]  # [bsz, 1, num_heads, head_dim]
        tmp_key_states = key_states  # [bsz, seq_len, num_heads, head_dim]

        tmp_query_states = tmp_query_states.permute(
            0, 2, 1, 3
        )  # [bsz, num_heads, 1, head_dim]
        tmp_key_states = tmp_key_states.permute(
            0, 2, 3, 1
        )  # [bsz, num_heads, head_dim, seq_len]

        attn_weights = torch.matmul(tmp_query_states, tmp_key_states) / math.sqrt(
            self.head_dim
        )  # [bsz, num_heads, 1, seq_len]
    else:
        attn_weights = None

    ######################################################################

    return attn_output, attn_weights, past_key_value


def _merge_power_max_last_calibrated(
    num_tokens_to_reduce, reduction_mask, metric, **kwargs
):
    device = metric.device

    significance_weights = metric[:, :, -1, :].max(dim=1).values

    position_ids = kwargs["position_ids"].clone()

    last_token_position = position_ids[:, -1].item()
    position_ids -= last_token_position
    position_ids *= -1
    position_ids = position_ids.to(torch.int).to(device)

    # TODO: Load bias to device
    bias_wrt_rel_pos = kwargs["bias"][kwargs["layer_idx"]].to(device)
    bias = bias_wrt_rel_pos[position_ids]

    significance_weights -= bias

    # Apply mask
    significance_weights = significance_weights.masked_fill(
        torch.logical_not(reduction_mask), math.inf
    )

    # Find indices to remove
    indices_to_remove = torch.topk(
        significance_weights, num_tokens_to_reduce, largest=False, dim=-1
    ).indices.squeeze()

    return indices_to_remove, significance_weights


# Class method for LlamaDecoderLayer
def _apply_pruning(self, hidden_states, metric):
    assert hidden_states.size(0) == 1, "HOMER does not support batch size > 1"

    # Update this dictionary to add new reduction modes
    reducers = {
        "power_max_last_calibrated": _merge_power_max_last_calibrated,
    }
    reducer = reducers.get(self.merge_manager.reduction_mode, None)

    if reducer is None:
        raise ValueError(f"Unknown reduction mode: {self.merge_manager.reduction_mode}")

    layer_reduction_info = self.merge_manager.layer_reduction_info

    # Retrieve merging variables
    num_tokens_to_reduce = layer_reduction_info["num_tokens_to_reduce"]
    reduction_mask = layer_reduction_info["reduction_mask"].to(hidden_states.device)

    batch_size, seq_len, _ = hidden_states.shape

    if num_tokens_to_reduce > 0:
        indices_to_remove, significance_weights = reducer(
            num_tokens_to_reduce,
            reduction_mask,
            metric,
            seq_len=seq_len,
            position_ids=layer_reduction_info["position_ids"],
            layer_idx=self.layer_idx,
            bias=self.merge_manager.bias,
        )

        # print(significance_weights)

        # Remove tokens
        prune_mask = torch.ones(hidden_states.size(1), dtype=torch.bool)
        prune_mask[indices_to_remove] = False
        hidden_states = hidden_states[:, prune_mask, :]

        self.merge_manager.set_layer_reduction_results(
            position_ids=layer_reduction_info["position_ids"][:, prune_mask],
            prune_mask=prune_mask,
            significance_weights=significance_weights,
        )

    else:
        self.merge_manager.set_layer_reduction_results(
            position_ids=layer_reduction_info["position_ids"],
            prune_mask=torch.ones(seq_len, dtype=torch.bool),
            significance_weights=torch.ones((batch_size, seq_len), dtype=metric.dtype),
        )

    return hidden_states


# Class method for LlamaDecoderLayer
def llamadecoderlayer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`, *optional*):
            attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
            query_sequence_length, key_sequence_length)` if default attention is used.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
            (see `past_key_values`).
        past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence
        kwargs (`dict`, *optional*):
            Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
            into the model
    """
    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
    )
    hidden_states = residual + hidden_states

    #################### MODIFIED FROM ORIGINAL CODE #####################

    # Token reduction
    if (
        hasattr(self, "merge_manager")
        and self.merge_manager.layer_reduction_info is not None
    ):
        hidden_states = self._apply_pruning(
            hidden_states=hidden_states,
            metric=self_attn_weights,
        )

    ######################################################################

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs


# Class method for LlamaForCausalLM
def __forward_and_reduce(
    self,
    chunk: Chunk,
    layers,
    target_len,
):
    """
    Forward & prune preprocessed/merged chunk.
    """

    # Hidden states of the most recent layer
    hidden_states = chunk.hidden_states

    position_ids = chunk.position_ids
    reduction_mask = chunk.reduction_mask
    past_key_value = chunk.cache
    num_tokens_to_reduce = hidden_states.shape[1] - target_len

    prune_mask = None
    significance_weights = None

    # Perform forward pass & token reduction
    for i, i_layer in enumerate(layers):
        if num_tokens_to_reduce > 0 and i == len(layers) - 1:
            self.merge_manager.set_layer_reduction_info(
                num_tokens_to_reduce=num_tokens_to_reduce,
                reduction_mask=reduction_mask,
                position_ids=position_ids,
            )
        else:
            self.merge_manager.set_layer_reduction_info()

        # self.merge_manager.set_block_reduction_info(block_reduction_info)

        # TODO: Implement decoder layer forward()
        decoder_outputs = self.model.layers[i_layer](
            hidden_states=hidden_states,
            attention_mask=None,  # Attention mask not required for Flash Attention
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=True,
            use_cache=True,
        )

        hidden_states = decoder_outputs[0]

        if num_tokens_to_reduce > 0 and i == len(layers) - 1:
            prune_mask = self.merge_manager.layer_reduction_results["prune_mask"]
            significance_weights = self.merge_manager.layer_reduction_results[
                "significance_weights"
            ]

    # TODO: Implement chunk prune()
    chunk.prune(prune_mask=prune_mask)
    chunk.update_hidden_states(hidden_states)

    # Set visualization information
    chunk.set_visualization_info(
        prune_mask=prune_mask,
        sig_weight=significance_weights,
    )

    torch.cuda.empty_cache()

    return chunk


# Class method for LlamaForCausalLM
def __process_leaf(
    self,
    chunk: Chunk,
    height,
    target_len=-1,
):
    """
    Change input_ids into embeddings, then pass the results to __forward_and_reduce()
    """
    assert height == 0
    assert target_len > 0

    input_ids = chunk.hidden_states

    assert input_ids is not None, "input_ids must be provided"

    # Step 1: Pass input_ids through the embedding layer
    batch_size, seq_length = input_ids.size()
    assert batch_size == 1
    input_ids = input_ids.view(-1, seq_length)

    inputs_embeds = self.model.embed_tokens(input_ids)
    hidden_states = inputs_embeds  # (layers, batch_size, seq_length, hidden_size)

    chunk.hidden_states = hidden_states  # Update chunk

    # Step 2: Get encoder layers for processing this chunk
    encoder_layers = torch.arange(
        self.merge_manager.layers_per_chunk
        + self.merge_manager.layers_leftover
        + self.merge_manager.layers_warmup
    )

    # Step 3: Process the chunk
    processed_chunk = self.__forward_and_reduce(
        chunk=chunk,
        layers=encoder_layers,
        target_len=target_len,
    )

    return processed_chunk


# Class method for LlamaForCausalLM
def __process_nonleaf(
    self,
    l_chunk,
    r_chunk,
    height,
    target_len=-1,
):
    """
    Merge the children chunks, then pass the results to __forward_and_reduce()
    """
    assert target_len > 0

    # Step 1: Merge children chunks
    merged_chunk = Chunk.merge(l_chunk, r_chunk)

    # Step 2: Get encoder layers for processing the merged chunk
    encoder_layers = (
        torch.arange(self.merge_manager.layers_per_chunk)
        + self.merge_manager.layers_leftover
        + self.merge_manager.layers_warmup
        + self.merge_manager.layers_per_chunk * height
    )

    # Step 3: Process the merged chunk
    processed_chunk = self.__forward_and_reduce(
        chunk=merged_chunk,
        layers=encoder_layers,
        target_len=target_len,
    )

    return processed_chunk


# Class method for LlamaForCausalLM
def _merge(
    self,
    chunks: List[Chunk],
    height,
    target_len=-1,
):
    """
    Perform recursive forward & merge of chunks
    """
    assert chunks[0].hidden_states is not None

    num_chunks = len(chunks)
    is_leaf = num_chunks == 1

    # Leaf node: Merge tokens to be size 1/2 with (layers_per_chunk + layers_leftover) layers
    if is_leaf:
        assert height == 0
        assert chunks[0] is not None

        return self.__process_leaf(
            chunk=chunks[0],
            height=height,
            target_len=target_len,
        )

    # Non-leaf: Perform recursive merge
    left_chunks = chunks[: num_chunks // 2]
    right_chunks = chunks[num_chunks // 2 :]

    # Options for recursive merge
    not_reduce_left = right_chunks[0] is None

    if not_reduce_left:
        l_chunk = self._merge(
            chunks=left_chunks,
            height=height - 1,
            target_len=self.merge_manager.max_chunk_len,  # If seqlen is shorter, no reduction is done thanks to __get_reduction_plan() function
        )
        r_chunk = None
    else:
        default_len = (  # This design is for backward compatibility (l_target_len & r_target_len used to be identical)
            self.merge_manager.prefix_len
            + self.merge_manager.eff_max_chunk_len // 2
            + self.merge_manager.suffix_len
        )
        if right_chunks[0].sequence_len < default_len:  # Right chunk is very short
            r_target_len = right_chunks[0].sequence_len
            l_target_len = self.merge_manager.max_chunk_len - (
                r_target_len
                - self.merge_manager.prefix_len
                - self.merge_manager.suffix_len
            )
        else:
            r_target_len = default_len
            l_target_len = default_len

        l_chunk = self._merge(
            chunks=left_chunks,
            height=height - 1,
            target_len=l_target_len,
        )
        r_chunk = self._merge(
            chunks=right_chunks,
            height=height - 1,
            target_len=r_target_len,
        )

    return self.__process_nonleaf(
        l_chunk=l_chunk,
        r_chunk=r_chunk,
        height=height,
        target_len=target_len,
    )


# Class method for LlamaForCausalLM
def create_homer_prefix(self, prefix_ids, context_ids, suffix_ids):
    assert prefix_ids is not None
    assert context_ids is not None
    assert suffix_ids is not None

    prefix_len = prefix_ids.size(1)
    context_len = context_ids.size(1)
    suffix_len = suffix_ids.size(1)

    total_len = prefix_len + suffix_len + context_len

    assert suffix_len > 0, "Suffix length must be greater than 0"

    # Compute "effective" chunk lengths (lengths without affix)
    eff_max_chunk_len = self.merge_manager.max_chunk_len - (prefix_len + suffix_len)

    # Derive a merging schedule
    num_merging_layers = len(self.model.layers) - self.merge_manager.layers_warmup
    if self.merge_manager.max_initial_chunk_len > 0:
        eff_max_initial_chunk_len = self.merge_manager.max_initial_chunk_len - (
            prefix_len + suffix_len
        )
        num_chunks = math.ceil(context_len / eff_max_initial_chunk_len)
    else:
        num_chunks = math.ceil(context_len / eff_max_chunk_len)

    tree_height = math.ceil(math.log2(num_chunks))

    num_chunks = (
        2**tree_height
    )  # num_chunks are power of 2 (to apply hierarchical context merging)
    eff_chunk_len = math.ceil(context_len / num_chunks)

    layers_per_chunk = math.floor(num_merging_layers / (tree_height + 1))
    layers_leftover = num_merging_layers - layers_per_chunk * (tree_height + 1)

    self.merge_manager.set_sample_info(
        prefix_len,
        suffix_len,
        eff_max_chunk_len,
        layers_per_chunk,
        layers_leftover,
    )

    chunks = Chunk.make_chunks(
        prefix_ids,
        context_ids,
        suffix_ids,
        num_chunks=num_chunks,
        eff_chunk_len=eff_chunk_len,
        visualize=self.merge_manager.visualize,
    )

    final_target_len = min(self.merge_manager.target_len, total_len)

    # Recursively merge the chunks
    chunk = self._merge(
        chunks=chunks,
        height=tree_height,
        target_len=final_target_len,
    )

    # Set visualization info
    self.visualization_info = chunk.get_visualization_info()
    self.context_ids = context_ids[0].cpu()

    assert self.config.pretraining_tp <= 1
    hidden_states = self.model.norm(chunk.hidden_states)
    logits = self.lm_head(hidden_states)
    logits = logits.float()

    # Reset merge_manager states
    self.merge_manager.set_layer_reduction_info()

    return {
        "cache": chunk.cache,
        "last_position_id": chunk.position_ids[0, -1],
        "logits": logits,
    }


def prepare_inputs_for_generation(
    self,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    cache_position=None,
    position_ids=None,
    use_cache=True,
    **kwargs,
):
    # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
    # Exception 1: when passing input_embeds, input_ids may be missing entries
    # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
    if past_key_values is not None:
        if inputs_embeds is not None:  # Exception 1
            input_ids = input_ids[:, -cache_position.shape[0] :]
        elif (
            input_ids.shape[1] != cache_position.shape[0]
        ):  # Default case (the "else", a no op, is Exception 2)
            input_ids = input_ids[:, cache_position]

    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1] :]

            #################### MODIFIED FROM ORIGINAL CODE #####################

            if hasattr(past_key_values, "pos_diff"):
                position_ids = position_ids - past_key_values.pos_diff

            ######################################################################

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and cache_position[0] == 0:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {
            "input_ids": input_ids.contiguous()
        }  # `contiguous()` needed for compilation use cases

    model_inputs.update(
        {
            "position_ids": position_ids,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "attention_mask": attention_mask,
        }
    )
    return model_inputs


def make_llamaforcausallm_forward(original_forward):
    def forward(self, homer_prefix=None, **kwargs):
        if homer_prefix is not None:
            input_ids = kwargs.get("input_ids", None)
            past_key_values = kwargs.get("past_key_values", None)
            position_ids = kwargs.get("position_ids", None)

            assert input_ids.size(0) == 1, "HOMER only support batch size 1"
            assert (
                past_key_values is None
            ), "past_key_values cannot be used with homer_prefix"

            if position_ids is None:
                position_ids = torch.arange(
                    input_ids.shape[1], device=input_ids.device
                ).unsqueeze(0)

            kwargs["position_ids"] = position_ids + homer_prefix["last_position_id"] + 1
            kwargs["past_key_values"] = homer_prefix["cache"]
            kwargs["use_cache"] = True

        return original_forward(**kwargs)

    return forward


def make_llamaforcausallm_generate(original_generate):
    def generate(self, homer_prefix=None, **kwargs):
        if homer_prefix is not None:
            assert (
                "input_ids" not in kwargs
            ), "input_ids should not be used together with homer_prefix"

            batch_size, _, prefix_len, _ = homer_prefix["cache"].key_cache[0].shape

            # Get last token's output
            new_token_idx = homer_prefix["logits"][0, -1].argmax()
            new_input_ids = new_token_idx.unsqueeze(0).unsqueeze(0).to(self.device)

            # Used in prepare_inputs_for_generation() to adjust position ids
            homer_prefix["cache"].pos_diff = (
                prefix_len - homer_prefix["last_position_id"]
            )

            input_ids = torch.cat(
                [
                    torch.ones((batch_size, prefix_len), device=self.device).long(),
                    new_input_ids,
                ],
                dim=-1,
            )

            kwargs["use_cache"] = True
            kwargs["attention_mask"] = torch.ones_like(input_ids)

            return original_generate(
                input_ids=input_ids, past_key_values=homer_prefix["cache"], **kwargs
            )

        return original_generate(**kwargs)

    return generate


def patch(
    model,
    max_chunk_len=2048,
    max_initial_chunk_len=-1,
    reduction_mode="power_max_last_calibrated",
    layers_warmup=0,
    target_len=2048,
    bias_path=None,
    visualize=False,
):
    """
    Args:
        model: LlamaForCausalLM
            Model to be patched
        max_chunk_len: int
            Maximum size for a single chunk
        max_initial_chunk_len: int
            Maximum number of tokens in the initial chunk
        reduction_mode: str
            Token reduction method to use
        layers_warmup: int
            Number of layers to use for warmup (i.e. no merging)
        target_len: int
            Final cache length
        bias_path: str
            Path to bias file which is used for calibration
        visualize: bool
            Store values for visualization (visualization not supported in current version)
    """
    assert isinstance(model, LlamaForCausalLM), f"Unsupported model type: {type(model)}"

    merge_manager = MergeManager(
        max_chunk_len=max_chunk_len,
        max_initial_chunk_len=max_initial_chunk_len,
        reduction_mode=reduction_mode,
        layers_warmup=layers_warmup,
        target_len=target_len,
        bias_path=bias_path,
        visualize=visualize,
    )

    model.create_homer_prefix = types.MethodType(create_homer_prefix, model)
    model._merge = types.MethodType(_merge, model)
    model.__process_nonleaf = types.MethodType(__process_nonleaf, model)
    model.__process_leaf = types.MethodType(__process_leaf, model)
    model.__forward_and_reduce = types.MethodType(__forward_and_reduce, model)
    model.prepare_inputs_for_generation = types.MethodType(
        prepare_inputs_for_generation, model
    )
    model.forward = types.MethodType(
        make_llamaforcausallm_forward(model.forward), model
    )
    model.generate = types.MethodType(
        make_llamaforcausallm_generate(model.generate), model
    )

    model.merge_manager = merge_manager

    for idx, layer in enumerate(model.model.layers):
        layer.layer_idx = idx
        layer.merge_manager = merge_manager

        layer.forward = types.MethodType(llamadecoderlayer_forward, layer)
        layer._apply_pruning = types.MethodType(_apply_pruning, layer)

        if isinstance(layer.self_attn, LlamaFlashAttention2):
            layer.self_attn.forward = types.MethodType(
                llamaflashattention2_forward, layer.self_attn
            )
