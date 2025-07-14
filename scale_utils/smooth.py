import torch
import torch.nn as nn

from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.bloom.modeling_bloom import BloomBlock
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, Qwen3DecoderLayer
from transformers.models.mistral.modeling_mistral import (
    MistralDecoderLayer,
    MistralRMSNorm,
)
from transformers.models.mixtral.modeling_mixtral import (
    MixtralDecoderLayer,
    MixtralRMSNorm,
)
from transformers.models.falcon.modeling_falcon import FalconDecoderLayer
from scale_utils.model_utils import RMSN
from mx import Linear


@torch.no_grad()
def smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    assert isinstance(ln, (nn.LayerNorm, RMSN))
    for fc in fcs:
        assert isinstance(fc, (nn.Linear, Linear))
        assert ln.weight.numel() == fc.in_features == act_scales.numel()

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)

    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )

    ln.weight.div_(scales)
    ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))


@torch.no_grad()
def smooth_ln_fcs_llama_like(ln, fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    assert isinstance(ln, (LlamaRMSNorm,
                           MistralRMSNorm,
                           MixtralRMSNorm,
                           Qwen2RMSNorm,
                           Qwen3RMSNorm,
                           RMSN))
    for fc in fcs:
        assert isinstance(fc, (nn.Linear, Linear))
        assert ln.weight.numel() == fc.in_features == act_scales.numel()
    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)
    # scales = (
    #     (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
    #     .clamp(min=1e-5)
    #     .to(device)
    #     .to(dtype)
    # )
    weight_scales = weight_scales.reshape(32, -1)
    max_per_group = weight_scales.abs().max(dim=0, keepdim=True)[0]
    weight_scales = max_per_group.expand_as(weight_scales).t().reshape(-1)
    scales = 2 ** (torch.log2(act_scales * weight_scales) // 2)

    ln.weight.div_(scales)
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))


@torch.no_grad()
def smooth_fcs(fc1, fc2, act_scales, alpha, head_dim=None,
               num_key_value_heads=None):
    assert isinstance(fc1, (nn.Linear, Linear))
    assert isinstance(fc2, (nn.Linear, Linear))
    device, dtype = fc1.weight.device, fc1.weight.dtype
    weight_scales = [fc1.weight.abs().max(dim=1)[0]]
    n_reg = fc2.in_features // fc1.out_features
    act_scales = act_scales.to(device=device, dtype=dtype)
    if n_reg > 1:
        assert head_dim is not None
        assert num_key_value_heads is not None
        act_scales = act_scales.reshape(num_key_value_heads, n_reg, head_dim).max(dim=1)[0].reshape(-1)
        weight_scales.append(
            fc2.weight.abs().max(dim=0)[0].reshape(
                num_key_value_heads, n_reg, head_dim).max(dim=1)[0].reshape(-1)
        )
    else:
        weight_scales.append(fc2.weight.abs().max(dim=0)[0])

    weight_scales = torch.stack(weight_scales, dim=0).max(dim=0)[0].clamp(min=1e-5)
    # scales = (
    #     (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
    #     .clamp(min=1e-5)
    #     .to(device=fc1.weight.device, dtype=fc1.weight.dtype)
    # )
    weight_scales = weight_scales.reshape(32, -1)
    max_per_group = weight_scales.abs().max(dim=0, keepdim=True)[0]
    weight_scales = max_per_group.expand_as(weight_scales).t().reshape(-1)
    scales = 2 ** (torch.log2(act_scales * weight_scales) // 2)
    fc1.weight.div_(scales.unsqueeze(1))
    if fc1.bias is not None:
        fc1.bias.div_(scales)
    if n_reg > 1:
        scales = scales.reshape(num_key_value_heads, -1)[:, None,
                                                         :].expand(num_key_value_heads, n_reg, head_dim).reshape(-1)
    fc2.weight.mul_(scales)


@torch.no_grad()
def smooth_lm(model, scales, alpha=0.5):
    for name, module in model.named_modules():
        if isinstance(module, OPTDecoderLayer):
            attn_ln = module.self_attn_layer_norm
            qkv = [
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ]
            qkv_input_scales = scales[name + ".self_attn.q_proj"]
            smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)

            ffn_ln = module.final_layer_norm
            fc1 = module.fc1
            fc1_input_scales = scales[name + ".fc1"]
            smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)
        elif isinstance(module, BloomBlock):
            attn_ln = module.input_layernorm
            qkv = module.self_attention.query_key_value
            qkv_input_scales = scales[name + ".self_attention.query_key_value"]
            smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)

            ffn_ln = module.post_attention_layernorm
            fc1 = module.mlp.dense_h_to_4h
            fc1_input_scales = scales[name + ".mlp.dense_h_to_4h"]
            smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)
        elif isinstance(module, FalconDecoderLayer):
            qkv = module.self_attention.query_key_value
            qkv_input_scales = scales[name + ".self_attention.query_key_value"]
            fc1_input_scales = scales[name + ".mlp.dense_h_to_4h"]
            fc1 = module.mlp.dense_h_to_4h

            if (
                not module.config.new_decoder_architecture
                and module.config.parallel_attn
            ):
                attn_ln = module.input_layernorm
                smooth_ln_fcs(attn_ln, [qkv, fc1], qkv_input_scales, alpha)
            else:
                attn_ln = (
                    module.ln_attn
                    if module.config.new_decoder_architecture
                    else module.input_layernorm
                )
                ffn_ln = (
                    module.ln_mlp
                    if module.config.new_decoder_architecture
                    else module.post_attention_layernorm
                )
                smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)
                smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)
        elif isinstance(module, (LlamaDecoderLayer, MistralDecoderLayer,
                                 Qwen2DecoderLayer, Qwen3DecoderLayer)):
            attn_ln = module.input_layernorm  # attention forward norm
            qkv = [
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ]

            qkv_input_scales = scales[name + ".self_attn.q_proj"]
            smooth_ln_fcs_llama_like(attn_ln, qkv, qkv_input_scales, alpha)
            smooth_fcs(module.self_attn.v_proj, module.self_attn.o_proj,
                       scales[name + ".self_attn.v_proj"], alpha,
                       head_dim=module.self_attn.head_dim,
                       num_key_value_heads=module.self_attn.config.num_key_value_heads)
            ffn_ln = module.post_attention_layernorm  # feed forward norm
            fcs = [module.mlp.gate_proj, module.mlp.up_proj]
            fcs_input_scales = scales[name + ".mlp.gate_proj"]

            smooth_ln_fcs_llama_like(ffn_ln, fcs, fcs_input_scales, alpha)
            smooth_fcs(module.mlp.up_proj, module.mlp.down_proj,
                       scales[name + ".mlp.down_proj"], alpha)
        elif isinstance(module, MixtralDecoderLayer):
            attn_ln = module.input_layernorm  # attention forward norm
            qkv = [
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ]

            qkv_input_scales = scales[name + ".self_attn.q_proj"]
            smooth_ln_fcs_llama_like(attn_ln, qkv, qkv_input_scales, alpha)

            ffn_ln = module.post_attention_layernorm  # feed forward norm
            fcs = [module.block_sparse_moe.gate]
            for expert in module.block_sparse_moe.experts:
                fcs.append(expert.w1)
                fcs.append(expert.w3)
            fcs_input_scales = scales[name + ".block_sparse_moe.gate"]

            smooth_ln_fcs_llama_like(ffn_ln, fcs, fcs_input_scales, alpha)
