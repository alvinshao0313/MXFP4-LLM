# modeling_XX.py from transformers==4.36.2
# for replacing torch.matmul to self.matmul
import math
from typing import List, Optional, Tuple, Union, Callable

import torch
from torch import nn
from transformers.cache_utils import Cache, EncoderDecoderCache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from transformers.models.llama.modeling_llama import (
    rotate_half,
    apply_rotary_pos_emb,
    repeat_kv,
)


def _eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = module.matmul1(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = module.matmul1(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def llama_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attention_interface: Callable = _eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def opt_forward(
    self,
    hidden_states: torch.Tensor,
    past_key_value: Optional[tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    layer_head_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
    cache_position: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
    """Input shape: Batch x Time x Channel"""
    bsz, tgt_len, _ = hidden_states.size()

    # Scaling is susceptible to floating point arithmetics' inprecisions
    # which can lead to different results (this is dependent from model
    # to model, e.g. whisper is one such case). We therefore keep the
    # original order of scaling to follow the original implementation
    # and enforce no scaling (1.0) in the attention call below.
    query_states = self.q_proj(hidden_states) * self.scaling
    query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)

    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)

    if past_key_value is not None:
        # save all key/value_states to cache to be re-used for fast auto-regressive generation
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, {"cache_position": cache_position}
        )

    attention_interface: Callable = _eager_attention_forward

    if self.config._attn_implementation != "eager":
        if self.config._attn_implementation == "sdpa" and output_attentions:
            import logging
            logger = logging.get_logger(__name__)
            logger.warning_once(
                "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.dropout,
        scaling=1.0,
        **kwargs,
    )

    attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous()
    attn_output = self.out_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def qwen_attn(self, query, key, value, causal_mask=None, attention_mask=None, head_mask=None):
    device = query.device
    if self.use_cache_quantization:
        raise NotImplementedError
        qk, qk_scale, qk_zero = key
        if self.use_cache_kernel and self.cache_kernels is not None:
            shape = query.shape[:-1] + (qk.shape[-2],)
            attn_weights = torch.zeros(shape, dtype=torch.float16, device=device)
            self.cache_kernels.vecquant8matmul_batched_faster_old(
                query.contiguous() if query.dtype == torch.float16 else query.to(torch.float16).contiguous(),
                qk.transpose(-1, -2).contiguous(),
                attn_weights,
                qk_scale.contiguous() if qk_scale.dtype == torch.float16 else qk_scale.to(torch.float16).contiguous(),
                qk_zero.contiguous()if qk_zero.dtype == torch.float16 else qk_zero.to(torch.float16).contiguous())
            # attn_weights = attn_weights.to(query.dtype).contiguous()
        else:
            key = dequantize_cache_torch(qk, qk_scale, qk_zero)
            attn_weights = torch.matmul(query, key.transpose(-1, -2))
    else:
        # attn_weights = torch.matmul(query, key.transpose(-1, -2))
        attn_weights = self.matmul1(query, key.transpose(-1, -2))

    if self.scale_attn_weights:
        if self.use_cache_quantization:
            size_temp = value[0].size(-1)
        else:
            size_temp = value.size(-1)
        attn_weights = attn_weights / (size_temp ** 0.5)

    mask_value = torch.finfo(attn_weights.dtype).min
    if causal_mask is not None:
        attn_weights = torch.where(
            causal_mask, attn_weights.to(attn_weights.dtype), mask_value
        )

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    if self.softmax_in_fp32:
        attn_weights = nn.functional.softmax(attn_weights.float(), dim=-1)
    else:
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    attn_weights = attn_weights.type(query.dtype)
    attn_weights = self.attn_dropout(attn_weights)

    if head_mask is not None:
        attn_weights = attn_weights * head_mask

    if self.use_cache_quantization:
        raise NotImplementedError
        qv, qv_scale, qv_zero = value
        if self.use_cache_kernel and self.cache_kernels is not None:
            shape = attn_weights.shape[:-1] + (query.shape[-1],)
            attn_output = torch.zeros(shape, dtype=torch.float16, device=device)
            self.cache_kernels.vecquant8matmul_batched_column_compression_faster_old(
                attn_weights.contiguous() if attn_weights.dtype == torch.float16 else attn_weights.to(torch.float16).contiguous(),
                qv.contiguous(),  # dtype: int32
                attn_output,
                qv_scale.contiguous() if qv_scale.dtype == torch.float16 else qv_scale.to(torch.float16).contiguous(),
                qv_zero.contiguous() if qv_zero.dtype == torch.float16 else qv_zero.to(torch.float16).contiguous())
            if attn_output.dtype != query.dtype:
                attn_output = attn_output.to(query.dtype)
                attn_weights = attn_weights.to(query.dtype)
        else:
            value = dequantize_cache_torch(qv, qv_scale, qv_zero)
            attn_output = torch.matmul(attn_weights, value)
    else:
        # attn_output = torch.matmul(attn_weights, value)
        attn_output = self.matmul2(attn_weights, value)

    attn_output = attn_output.transpose(1, 2)

    return attn_output, attn_weights


def mistral_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attention_interface: Callable = _eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=getattr(self.config, "sliding_window", None),  # main diff with Llama
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def qwen23_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attention_interface: Callable = _eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,  # main diff with Llama
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def llm_pruner_llama_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

    def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
        gather_indices = position_ids[:, None, :, None]  # [bs, 1, seq_len, 1]
        gather_indices = gather_indices.repeat(1, cos.shape[1], 1, cos.shape[3])
        cos = torch.gather(cos.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
        sin = torch.gather(sin.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    # [bsz, nh, t, hd]

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
    attn_weights = self.matmul1(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz * self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask
        attn_weights = torch.max(attn_weights, torch.tensor(
            torch.finfo(attn_weights.dtype).min, device=attn_weights.device))

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    # attn_output = torch.matmul(attn_weights, value_states)
    attn_output = self.matmul2(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, -1)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def bart_forward(
    self,
    hidden_states: torch.Tensor,
    key_value_states: Optional[torch.Tensor] = None,
    past_key_value: Optional[Cache] = None,
    attention_mask: Optional[torch.Tensor] = None,
    layer_head_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
    cache_position: Optional[torch.Tensor] = None,
    # TODO: we need a refactor so that the different attention modules can get their specific kwargs
    # ATM, we have mixed things encoder, decoder, and encoder-decoder attn
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel"""

    # if key_value_states are provided this layer is used as a cross-attention layer
    # for the decoder
    is_cross_attention = key_value_states is not None

    # determine input shapes
    bsz, tgt_len = hidden_states.shape[:-1]
    src_len = key_value_states.shape[1] if is_cross_attention else tgt_len

    q_input_shape = (bsz, tgt_len, -1, self.head_dim)
    kv_input_shape = (bsz, src_len, -1, self.head_dim)

    # get query proj
    query_states = self.q_proj(hidden_states).view(*q_input_shape).transpose(1, 2)

    if past_key_value is not None:
        if isinstance(past_key_value, EncoderDecoderCache):
            is_updated = past_key_value.is_updated.get(self.layer_idx)
            if is_cross_attention:
                # after the first generated id, we can subsequently re-use all key/value_states from cache
                curr_past_key_value = past_key_value.cross_attention_cache
            else:
                curr_past_key_value = past_key_value.self_attention_cache
        else:
            curr_past_key_value = past_key_value

    current_states = key_value_states if is_cross_attention else hidden_states
    if is_cross_attention and past_key_value is not None and is_updated:
        # reuse k,v, cross_attentions
        key_states = curr_past_key_value.key_cache[self.layer_idx]
        value_states = curr_past_key_value.value_cache[self.layer_idx]
    else:
        key_states = self.k_proj(current_states)
        value_states = self.v_proj(current_states)
        key_states = key_states.view(*kv_input_shape).transpose(1, 2)
        value_states = value_states.view(*kv_input_shape).transpose(1, 2)

        if past_key_value is not None:
            # save all key/value_states to cache to be re-used for fast auto-regressive generation
            cache_position = cache_position if not is_cross_attention else None
            key_states, value_states = curr_past_key_value.update(
                key_states, value_states, self.layer_idx, {"cache_position": cache_position}
            )
            # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
            if is_cross_attention:
                past_key_value.is_updated[self.layer_idx] = True

    attention_interface: Callable = _eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.dropout,
        scaling=self.scaling,
        output_attentions=output_attentions,
        head_mask=layer_head_mask,
        **kwargs,
    )

    attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous()
    attn_output = self.out_proj(attn_output)

    return attn_output, attn_weights, past_key_value
