import torch

def llama_ln_fusion(model):
    device = model.device
    dtype = torch.float
    import copy
    llama_dict = model.state_dict()
    fused_dict = dict()
    for k, v in llama_dict.items():
        fused_dict[k] = v.data.cpu()
    i = 0
    
    for layer in tqdm(range(100)):
        if f"model.layers.{layer}.input_layernorm.weight" in llama_dict.keys():
            attn_ln_weight = llama_dict[f"model.layers.{layer}.input_layernorm.weight"].data
            ffn_ln_weight = llama_dict[f"model.layers.{layer}.post_attention_layernorm.weight"].data
            
            q_weight_fused = llama_dict[f"model.layers.{layer}.self_attn.q_proj.weight"].data * attn_ln_weight
            k_weight_fused = llama_dict[f"model.layers.{layer}.self_attn.k_proj.weight"].data * attn_ln_weight
            v_weight_fused = llama_dict[f"model.layers.{layer}.self_attn.v_proj.weight"].data * attn_ln_weight
        
            fc_1_weight_fused = llama_dict[f"model.layers.{layer}.mlp.up_proj.weight"].data * ffn_ln_weight
            fc_3_weight_fused = llama_dict[f"model.layers.{layer}.mlp.gate_proj.weight"].data * ffn_ln_weight
        
            # Fusing
            fused_dict[f"model.layers.{layer}.self_attn.q_proj.weight"].copy_(q_weight_fused.data)
            fused_dict[f"model.layers.{layer}.self_attn.k_proj.weight"].copy_(k_weight_fused.data)
            fused_dict[f"model.layers.{layer}.self_attn.v_proj.weight"].copy_(v_weight_fused.data)
            
            fused_dict[f"model.layers.{layer}.mlp.up_proj.weight"].copy_(fc_1_weight_fused.data)
            fused_dict[f"model.layers.{layer}.mlp.gate_proj.weight"].copy_(fc_3_weight_fused.data)
            
            # LN weight = 1
            fused_dict[f"model.layers.{layer}.input_layernorm.weight"] = torch.ones(attn_ln_weight.shape)
            fused_dict[f"model.layers.{layer}.post_attention_layernorm.weight"] = torch.ones(ffn_ln_weight.shape)
            i += 1
        else:
            continue

    for k, v in fused_dict.items():
        model.state_dict()[k].copy_(v.data)

    print(f'LLaMa fusion done (for {i} layers)')
    del fused_dict
    torch.cuda.empty_cache()
