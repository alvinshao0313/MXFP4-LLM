
import functools
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
import os
from scale_utils.model_utils import get_model
from utils import calib
from utils.common import *
try:
    from scale_utils import hadamard_utils
    import fast_hadamard_transform
except:
    print('hadamard_utils is not imported')
import math
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def get_act_scales(model, testloader):
    model.eval()
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    device = next(model.parameters()).device
    act_scales = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        # comming_mean = torch.mean(tensor, dim=0).float().cpu()
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max)
            # act_scales[name] = (act_scales[name] + comming_mean) / 2
        else:
            act_scales[name] = comming_max
            # act_scales[name] = comming_mean

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(functools.partial(stat_input_hook, name=name))
            )

    for batch in tqdm(testloader, desc="Calculating activation scales"):
        input_ids = batch[0].to(device)
        model(input_ids)

    for h in hooks:
        h.remove()

    return act_scales


def rotate_hook(rotate_mode):
    def hook(module, inp):
        # Hadamard transform (QuaRot)
        if rotate_mode.get("online_full_had", False):
            inp[0].data = hadamard_utils.matmul_hadU_cuda(
                inp[0].data, rotate_mode["had_K"], rotate_mode["K"])
        elif rotate_mode.get("online_partial_had", False):
            init_shape = inp[0].shape
            if rotate_mode["K"] == 1:
                inp[0].data = fast_hadamard_transform.hadamard_transform(
                    inp[0].data.reshape(-1, init_shape[-1] // rotate_mode["had_dim"],
                                        rotate_mode["had_dim"]).transpose(1, 2),
                    scale=1 /
                    math.sqrt(
                        init_shape[-1] // rotate_mode["had_dim"])
                ).transpose(1, 2)
            else:
                inp[0].data = (rotate_mode["had_K"].to(inp[0].dtype).to(inp[0].device) @ inp[0].data.reshape(-1,
                                                                                                             init_shape[-1] // rotate_mode["had_dim"], rotate_mode["had_dim"])) / math.sqrt(init_shape[-1] // rotate_mode["had_dim"])
            inp[0].data = inp[0].data.reshape(init_shape)
        elif rotate_mode.get("online_group_had", False):
            assert rotate_mode["had_dim"] > 0 and rotate_mode[
                "K"] == 1, "Group Hadamard transform requires had_dim > 0 and K == 1"
            # Group Hadamard transform
            init_shape = inp[0].shape
            inp[0].data = fast_hadamard_transform.hadamard_transform(
                inp[0].data.reshape(-1, init_shape[-1] //
                                    rotate_mode["had_dim"], rotate_mode["had_dim"]),
                scale=1 / math.sqrt(rotate_mode["had_dim"]))
            inp[0].data = inp[0].data.reshape(init_shape)
        return inp
    return hook


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="meta-llama/Meta-Llama-3-8B", help="model name"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./smooth_scales/",
        help="where to save the act scales",
    )
    parser.add_argument(
        "--calib_dataset",
        type=str,
        default="wikitext2",
        help="Name of the calibration dataset, e.g., wikitext2, ptb, c4",
    )
    parser.add_argument("--nsamples", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument('--per_rotate', type=str2bool, default=True)
    parser.add_argument('--rotate_mode', type=str, default='group_hadamard',
                        choices=['hadamard', 'group_hadamard', 'identity'])
    parser.add_argument('--online_partial_had', type=str2bool, default=False)
    parser.add_argument('--sorting_transform', type=str2path, default=None)
    parser.add_argument('--rotate_kv', type=str2bool, default=True)
    parser.add_argument('--group_rotate_kv', type=str2bool, default=False)
    parser.add_argument('--block_size_linear', type=int, default=32)
    args = parser.parse_args()
    return args


@torch.no_grad()
def main():
    args = parse_args()
    model = get_model(args.model)
    if args.per_rotate:
        from scale_utils import rotation_utils
        rotation_utils.fuse_layer_norms(model)
        rotation_utils.rotate_model(model, args)
        rotation_utils.cleanup_memory(verbos=True)
        rotate_handles = []
        if 'llama' in args.model or 'mistral' in args.model:
            for name, mod in model.named_modules():
                rotate_mode = {}
                if 'down_proj' in name:
                    if args.rotate_mode == 'hadamard':
                        had_K, K = hadamard_utils.get_hadK(
                            model.config.intermediate_size)
                        rotate_mode["online_full_had"] = True
                    elif args.rotate_mode == 'group_hadamard':
                        had_K, K = hadamard_utils.get_hadK(
                            args.block_size_linear)
                        rotate_mode["online_group_had"] = True
                        rotate_mode["had_dim"] = args.block_size_linear
                    rotate_mode["had_K"] = had_K
                    rotate_mode["K"] = K
                    rotate_handles.append(mod.register_forward_pre_hook(
                        rotate_hook(rotate_mode)))
                elif 'o_proj' in name and args.online_partial_had:
                    had_K, K = hadamard_utils.get_hadK(
                        model.config.num_attention_heads)
                    rotate_mode["online_partial_had"] = True
                    rotate_mode["had_K"] = had_K
                    rotate_mode["K"] = K
                    rotate_mode["had_dim"] = model.config.hidden_size // model.config.num_attention_heads
                    rotate_handles.append(mod.register_forward_pre_hook(
                        rotate_hook(rotate_mode)))
        else:
            raise NotImplementedError

        # KV Cache
        if args.rotate_kv:
            from scale_utils import model_utils, rotation_utils
            rope_function_name = model_utils.get_rope_function_name(model)
            layers = model_utils.get_layers(model)
            had_dim = args.block_size_matmul if args.group_rotate_kv else -1
            for layer in layers:
                rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                    layer.self_attn,
                    rope_function_name,
                    config=model.config,
                    had_dim=had_dim,
                )

    trainloader = calib.get_loaders(args.calib_dataset, nsamples=args.nsamples,
                                    model=args.model, eval_mode=False)
    act_scales = get_act_scales(model, trainloader)
    # 每组取最大值 比per-channel更好
    for name, scale in act_scales.items():
        scale = scale.reshape(32, -1)
        max_per_group = scale.abs().max(dim=0, keepdim=True)[0]
        scale = max_per_group.expand_as(scale).t()
        act_scales[name] = scale.reshape(-1)
    args.output_path = os.path.join(args.output_path, f"{args.model.split('/')[-1]}-post-group-smooth-scales.pt")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(act_scales, args.output_path)
    print(f"Activation scales saved to {args.output_path}")


if __name__ == "__main__":
    main()
