import torch
import os
import argparse
import torch.nn as nn
import transformers
import functools
from tqdm import tqdm
import gc
import sys
from scale_utils import model_utils
from scale_utils import rotation_utils
from utils import calib
from utils.common import *
import os
from transformers.cache_utils import DynamicCache
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
default_path = '/home/shaoyuantian/program/MXFP4-LLM/hidden_state_samples/'


def scatter_largest_to_groups(sorted_idx, block_size):
    N = sorted_idx.shape[-1]
    num_groups = N // block_size
    grouped_idx = torch.empty_like(sorted_idx)
    for i in range(N):
        group = i % num_groups
        pos_in_group = i // num_groups
        grouped_idx[group * block_size + pos_in_group] = sorted_idx[i]
    return grouped_idx


def get_sorting_index(model, dataloader, save_path, args):
    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = False
    config = model.config
    dtype = config.torch_dtype
    head_dim = config.hidden_size // config.num_attention_heads
    model_type = model_utils.model_type_extractor(model)
    dev = 'cuda'  # next(model.parameters()).device

    same_type = [model_utils.LLAMA_MODEL, model_utils.MISTRAL_MODEL, model_utils.QWEN2_MODEL]
    if model_type in same_type:
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        # for transformers >= 4.44.2,model.model has rotary emb
        if hasattr(model.model, "rotary_emb"):
            model.model.rotary_emb = model.model.rotary_emb.to(dev)
        layers = model.model.layers
    else:
        raise ValueError(
            "Model type not supported for training data extraction.")

    layers[0] = layers[0].to(dev)
    inps = torch.zeros(
        (args.nsamples, args.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0}

    # catch the first layer input   捕获第一层的输入
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs.get("position_ids", None)
            raise ValueError

    layers[0] = Catcher(layers[0])
    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass

    # move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if model_type in same_type:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    else:
        raise ValueError(
            "Model type not supported for training data extraction.")

    # input of first layer for fp model
    fp_inps = inps

    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    if attention_mask is not None:
        attention_mask = attention_mask.repeat(1, 1, 1, 1).to(dtype)
    else:
        attention_mask = None

    del inps, cache, dataloader, batch
    model.to('cpu')
    r1_act_avg = {'idx': 0,
                  'act_avg': torch.zeros((1, model.config.hidden_size), dtype=dtype, device=dev)}
    r2_act_avg = {}
    r3_act_avg = {'idx': 0,
                  'act_avg': torch.zeros((1, model.config.hidden_size), dtype=dtype, device=dev)}
    r4_act_avg = {}

    r1_blocks = ["self_attn.q_proj",
                 "mlp.up_proj",]
    r2_blocks = ["self_attn.o_proj",]
    r4_blocks = ["mlp.down_proj"]
    # q_states = {}
    # k_states = {}
    # past_key_value = DynamicCache()

    for idx, decoder_layer in tqdm(enumerate(layers), total=len(layers), desc="Processing layers"):
        decoder_layer = decoder_layer.to(dev)
        r2_act_avg[f'layer_{idx}_act_avg'] = torch.zeros(
            (1, model.config.hidden_size), dtype=dtype, device=dev)
        r2_act_avg[f'layer_{idx}_idx'] = 0
        r4_act_avg[f'layer_{idx}_act_avg'] = torch.zeros(
            (1, model.config.intermediate_size), dtype=dtype, device=dev)
        r4_act_avg[f'layer_{idx}_idx'] = 0

        def stat_tensor(name, tensor):
            t_num, hidden_size = tensor.view(-1, tensor.shape[-1]).shape
            if name in r1_blocks:
                torch.save(tensor, os.path.join(
                    save_path, f'{args.model.split("/")[-1]}-layer-{idx}-r1.pt'))
                return
                # r1_act_avg['act_avg'] = torch.maximum(
                #     r1_act_avg['act_avg'], tensor.view(-1, hidden_size).abs().max(dim=0, keepdim=True)[0])
                r1_act_avg['act_avg'] = (r1_act_avg['act_avg'] * r1_act_avg['idx'] +
                                         tensor.view(-1, hidden_size).sum(dim=0, keepdim=True)) / (r1_act_avg['idx'] + t_num)
                r1_act_avg['idx'] += t_num
            elif name in r2_blocks:
                torch.save(tensor, os.path.join(
                    save_path, f'{args.model.split("/")[-1]}-layer-{idx}-r2.pt'))
                return
                r2_act_avg[f'layer_{idx}_act_avg'] = (r2_act_avg[f'layer_{idx}_act_avg'] * r2_act_avg[f'layer_{idx}_idx'] +
                                                      tensor.view(-1, hidden_size).sum(dim=0, keepdim=True)) / (r2_act_avg[f'layer_{idx}_idx'] + t_num)
                r2_act_avg[f'layer_{idx}_idx'] += t_num
            elif name in r4_blocks:
                torch.save(tensor, os.path.join(
                    save_path, f'{args.model.split("/")[-1]}-layer-{idx}-r4.pt'))
                return
                r4_act_avg[f'layer_{idx}_act_avg'] = (r4_act_avg[f'layer_{idx}_act_avg'] * r4_act_avg[f'layer_{idx}_idx'] +
                                                      tensor.view(-1, hidden_size).sum(dim=0, keepdim=True)) / (r4_act_avg[f'layer_{idx}_idx'] + t_num)
                r4_act_avg[f'layer_{idx}_idx'] += t_num

        def stat_input_hook(m, x, y, name):
            if isinstance(x, tuple):
                x = x[0]
            stat_tensor(name, x)

        # 插入钩子函数
        hooks = []
        module_list = [
            'self_attn.q_proj',
            'self_attn.o_proj',
            'mlp.up_proj',
            'mlp.down_proj',
            # 'block_sparse_moe.gate',
            # 'mlp.shared_experts.up_proj',
        ]
        for name, m in decoder_layer.named_modules():
            if isinstance(m, nn.Linear) and name in module_list:
                hooks.append(
                    m.register_forward_hook(
                        functools.partial(stat_input_hook, name=name)))

        # 记录 layer idx 输出
        with torch.no_grad():
            for j in range(args.nsamples):
                layer_outputs = decoder_layer(fp_inps[j].unsqueeze(
                    0), attention_mask=attention_mask, position_ids=position_ids,
                    # use_cache=True,
                    # past_key_value=past_key_value
                )
                fp_inps[j] = layer_outputs[0]

        # k_states[f'layer_{idx}_act_avg'] = past_key_value.key_cache[idx].squeeze(
        # ).cpu().transpose(0, 1).reshape(-1, config.num_key_value_heads * head_dim)
        # k_states[f'layer_{idx}_act_avg'] = k_states[f'layer_{idx}_act_avg'].mean(dim=0, keepdim=True)
        # past_key_value.key_cache[idx] = 0
        # past_key_value.value_cache[idx] = 0

        for h in hooks:
            h.remove()

        decoder_layer = decoder_layer.to('cpu')

    sorted_idx = {}
    _, sorted_idx['R1'] = torch.sort(r1_act_avg['act_avg'].squeeze().abs(), dim=-1, descending=True)
    sorted_idx['R1'] = scatter_largest_to_groups(sorted_idx['R1'], 32)
    for i in range(len(layers)):
        r2_act_avg[f'layer_{i}_act_avg'] = r2_act_avg[f'layer_{i}_act_avg'].squeeze()
        r2_act_avg[f'layer_{i}_act_avg'] = r2_act_avg[f'layer_{i}_act_avg'].reshape(config.num_key_value_heads,
                                                                                    config.num_attention_heads // config.num_key_value_heads, -1)
        r2_act_avg[f'layer_{i}_act_avg'] = r2_act_avg[f'layer_{i}_act_avg'].mean(
            dim=1).reshape(config.num_key_value_heads, -1)
        _, sorted_idx[f"model.layers.{i}.self_attn.R2"] = torch.sort(
            r2_act_avg[f'layer_{i}_act_avg'].abs(), dim=-1, descending=True)
        for j in range(config.num_key_value_heads):
            sorted_idx[f"model.layers.{i}.self_attn.R2"][j] = scatter_largest_to_groups(
                sorted_idx[f"model.layers.{i}.self_attn.R2"][j], 32)
            sorted_idx[f"model.layers.{i}.self_attn.R2"][j] = sorted_idx[f"model.layers.{i}.self_attn.R2"][j] + \
                j * config.hidden_size // config.num_attention_heads
        sorted_idx[f"model.layers.{i}.self_attn.R2"] = sorted_idx[f"model.layers.{i}.self_attn.R2"].reshape(-1)
        _, sorted_idx[f"model.layers.{i}.self_attn.R4"] = torch.sort(
            r4_act_avg[f'layer_{i}_act_avg'].squeeze().abs(), dim=-1, descending=True)
        sorted_idx[f"model.layers.{i}.self_attn.R4"] = scatter_largest_to_groups(
            sorted_idx[f"model.layers.{i}.self_attn.R4"], 32)
    torch.save(sorted_idx, os.path.join(
        save_path, f'{args.model.split("/")[-1]}-mean-sorted-idx.pt'))
    print(f'Saved sorted index.')
    # torch.save(q_states, os.path.join(save_path, f'{args.model.split("/")[-1]}-q_states.pt'))
    # torch.save(k_states, os.path.join(save_path, f'{args.model.split("/")[-1]}-k_states.pt'))

    del fp_inps
    torch.cuda.empty_cache()
    gc.collect()
    model.config.use_cache = use_cache

    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Meta-Llama-3-70B',
                        help='model name')
    parser.add_argument('--hf_token', type=str, default=None)
    parser.add_argument('--r_path', type=str, default='',
                        help='where to save the r1&2&4 training data')
    parser.add_argument("--calib_dataset", type=str, default="wikitext2",
                        choices=["wikitext2", "ptb", "c4"],
                        help="Where to extract calibration data from.",)
    parser.add_argument('--per_smooth', type=str2path, default=False,
                        help='Whether to apply per-smooth on the model.')
    parser.add_argument('--nsamples', type=int, default=1)
    parser.add_argument('--seqlen', type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0, help="Seed for sampling the calibration data.")
    args = parser.parse_args()

    if args.r_path == '':
        args.r_path = default_path

    if not os.path.exists(args.r_path):
        os.makedirs(args.r_path)
    print(f"---> r_path: {args.r_path}")
    return args


@torch.no_grad()
def main():
    args = parse_args()
    transformers.set_seed(args.seed)
    model = model_utils.get_model(args.model, args.hf_token)
    if args.per_smooth:
        from scale_utils import smooth
        smooth_scales = torch.load(args.per_smooth, weights_only=False)
        smooth.smooth_lm(model, smooth_scales, 1.0)
    rotation_utils.fuse_layer_norms(model)
    dataloader = calib.get_loaders(
        args.calib_dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model=args.model,
        seqlen=args.seqlen,
        eval_mode=False
    )
    get_sorting_index(model=model,
                      dataloader=dataloader,
                      save_path=args.r_path,
                      args=args)


if __name__ == '__main__':
    main()
