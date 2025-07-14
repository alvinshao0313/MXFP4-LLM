# AIHA 2024
# Reduced-precision LLM inference framework with various precision
# Quantization framework is based on MX-format (https://github.com/microsoft/microxcaling)

import os
import gc
import time
import json
import torch
import string
import random
from datetime import datetime
import argparse
import numpy as np
import transformers
import torch.nn as nn
from tqdm import tqdm
from transformers import TextStreamer
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.common import *
from utils.mx import *
import lm_eval
from lm_eval import evaluator
import warnings
import logging
import pprint
import math
from accelerate import Accelerator
warnings.filterwarnings('ignore')

try:
    from scale_utils import hadamard_utils
    import fast_hadamard_transform
except:
    print('hadamard_utils is not imported')


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


def main(args):
    # ============================ Load model
    # QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs (arXiv:2404.00456)
    if args.rotate:
        logging.info(f"Applying Hadamard transform")
        kwargs = {'device_map': 'cpu', 'trust_remote_code': True,
                  'attn_implementation': "eager"}
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype='auto' if args.auto_dtype else torch.bfloat16, **kwargs)
        from scale_utils import rotation_utils, gptq_utils
        rotation_utils.fuse_layer_norms(model)
        rotation_utils.rotate_model(model, args)
        rotation_utils.cleanup_memory(verbos=True)
        if args.gptq:
            from utils import calib
            trainloader = calib.get_loaders(args.gptq_cal_dataset, nsamples=args.gptq_cal_nsamples,
                                            model=args.model, eval_mode=False)
            gptq_utils.gptq_fwrd(model, trainloader, 'cuda:0', args)
    else:
        dev = 'balanced'
        kwargs = {'device_map': dev, 'trust_remote_code': True,
                  'attn_implementation': "eager"}
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype='auto' if args.auto_dtype else torch.float32, **kwargs)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, use_fast=False)

    # ============================ MX format
    # mx_specs_linear = parse_mx_specs(args, 'linear')
    # mx_specs_matmul = parse_mx_specs(args, 'matmul')
    # mx_specs_ln = parse_mx_specs(args, 'ln')
    # mx_specs_head = parse_mx_specs(args, 'head')
    # get_mx_model(
    #     model.eval(),
    #     mx_specs_linear=mx_specs_linear,
    #     mx_specs_matmul=mx_specs_matmul,
    #     mx_specs_ln=mx_specs_ln,
    #     mx_specs_head=mx_specs_head,
    #     args=args,
    # )
    # gptq_mx_specs_linear = parse_mx_specs(args, 'linear')
    # gptq_mx_specs_linear

    # ============================ Runtime Hadamard Transform for QuaRot
    if args.rotate:
        from scale_utils import hadamard_utils
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

    # Load into GPU
    if model.device.type == 'cpu':
        accelerator = Accelerator()
        model = accelerator.prepare(model)

    # ============================ Evaluation
    if args.eval_ppl:
        seqlen = 2048  # hard-coding
        args.limit = -1  # whole samples
        if 'llama3' in args.model:
            cache_testloader = f'calibset/wikitext_test_{seqlen}_{args.seed}_llama3.cache'
        elif 'llama2' in args.model:
            cache_testloader = f'calibset/wikitext_test_{seqlen}_{args.seed}_llama2.cache'
        elif 'llama' in args.model:
            cache_testloader = f'calibset/wikitext_test_{seqlen}_{args.seed}_llama.cache'
        elif 'mistral' in args.model:
            cache_testloader = f'calibset/wikitext_test_{seqlen}_{args.seed}_mistral.cache'
        elif 'qwen2' in args.model:
            cache_testloader = f'calibset/wikitext_test_{seqlen}_{args.seed}_qwen2.cache'
        elif 'qwen' in args.model:
            cache_testloader = f'calibset/wikitext_test_{seqlen}_{args.seed}_qwen.cache'
        elif 'gpt' in args.model:
            cache_testloader = f'calibset/wikitext_test_{seqlen}_{args.seed}_gpt.cache'
        elif 'opt' in args.model:
            cache_testloader = f'calibset/wikitext_test_{seqlen}_{args.seed}_opt.cache'
        elif 'midm' in args.model:
            cache_testloader = f'calibset/wikitext_test_{seqlen}_{args.seed}_midm.cache'
        else:
            cache_testloader = f'calibset/wikitext_test_{seqlen}_{args.seed}.cache'
        if os.path.exists(cache_testloader):
            testloader = torch.load(cache_testloader, weights_only=False)
            logging.info(f"load calibration from {cache_testloader}")
        else:
            from utils.calib import get_wikitext2_test
            testloader = get_wikitext2_test(
                seed=args.seed, seqlen=seqlen, model=args.model)
            if not os.path.exists('calibset'):
                os.mkdir('calibset')
            torch.save(testloader, cache_testloader)
        testenc = testloader.input_ids
        nsamples = testenc.numel() // seqlen
        use_cache = model.config.use_cache
        model.config.use_cache = False
        model.eval()
        nlls = []
        with torch.no_grad():
            pbar = tqdm(range(nsamples))
            for i in pbar:
                batch = testenc[:, (i * seqlen): ((i + 1) * seqlen)].to('cuda')
                if "opt" in args.model.lower():
                    outputs = model.model.decoder(batch)
                elif "llama" in args.model.lower() or "mixtral" in args.model.lower():
                    outputs = model.model(batch)
                elif "mistral" in args.model.lower():
                    outputs = model.model(batch)
                elif "qwen2" in args.model.lower():
                    outputs = model.model(batch)
                elif "qwen" in args.model.lower():
                    outputs = model.transformer(batch)
                elif "falcon" in args.model:
                    outputs = model.transformer(batch)
                hidden_states = outputs[0]
                logits = model.lm_head(hidden_states)
                shift_logits = logits[:, :-1, :]
                shift_labels = testenc[:, (i * seqlen): ((i + 1) * seqlen)][
                    :, 1:
                ].to(model.lm_head.weight.device)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                neg_log_likelihood = loss.float() * seqlen
                nlls.append(neg_log_likelihood)
                # pbar.set_description(f'loss: {loss.item():.4f}')
                ppl = torch.exp(torch.stack(nlls).mean() / seqlen)
                pbar.set_description(f"PPL: {ppl.item():.2f}")
                if i == args.limit:
                    break

            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
        logging.info(f'wikitext ppl : {ppl.item()}')
        model.config.use_cache = use_cache
        results = {'wiki_ppl': ppl.item()}

    if args.tasks:  # lm-eval
        lm = lm_eval.models.huggingface.HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            backend='causal',
            trust_remote_code=True,
            batch_size="auto",
        )

        with torch.no_grad():
            results = evaluator.simple_evaluate(
                model=lm,
                tasks=args.tasks,
                num_fewshot=args.num_fewshot,
                batch_size="auto"
            )
        results = results['results']
        logging.info(pprint.pformat(results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model and Datsets
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tasks', type=str2list, default=[])
    parser.add_argument('--num_fewshot', type=str2int, default='none')
    parser.add_argument('--eval_ppl', type=str2bool, default=False)
    # Bit-configuration (Linear)
    parser.add_argument('--w_elem_format_linear', type=str, default='fp6_e3m2')
    parser.add_argument('--a_elem_format_linear', type=str, default='fp4_e2m1')
    parser.add_argument('--scale_bits_linear', type=int, default=8)
    parser.add_argument('--block_size_linear', type=int, default=32)
    # Bit-configuration (MatMul)
    parser.add_argument('--A_elem_format_matmul', type=str, default='fp6_e3m2')
    parser.add_argument('--B_elem_format_matmul', type=str, default='fp4_e2m1')
    parser.add_argument('--scale_bits_matmul', type=int, default=8)
    parser.add_argument('--block_size_matmul', type=int, default=32)
    # Bit-configuration (LayerNorm)
    parser.add_argument('--w_elem_format_ln', type=str, default='fp6_e3m2')
    parser.add_argument('--a_elem_format_ln', type=str, default='fp6_e3m2')
    parser.add_argument('--scale_bits_ln', type=int, default=8)
    parser.add_argument('--block_size_ln', type=int, default=32)
    # Bit-configuration (LM-Head)
    parser.add_argument('--w_elem_format_head', type=str, default='fp6_e3m2')
    parser.add_argument('--a_elem_format_head', type=str, default='fp6_e3m2')
    parser.add_argument('--scale_bits_head', type=int, default=8)
    parser.add_argument('--block_size_head', type=int, default=32)
    # Others
    parser.add_argument('--auto_dtype', type=str2bool, default=True)
    parser.add_argument('--custom_cuda', type=str2bool, default=False)
    parser.add_argument('--a_scale_mode', type=int, default=0)
    parser.add_argument('--w_scale_mode', type=int, default=0)
    parser.add_argument('--A_scale_mode', type=int, default=0)
    parser.add_argument('--B_scale_mode', type=int, default=0)
    parser.add_argument('--per_tensor', type=str2bool, default=False)
    # Weight Scaling
    parser.add_argument('--rotate', type=str2bool, default=False)
    parser.add_argument('--rotate_mode', type=str, default='hadamard',
                        choices=['hadamard', 'group_hadamard'])
    parser.add_argument('--online_partial_had', type=str2bool, default=False)
    parser.add_argument('--sorting_transform', type=str2path, default=None)
    parser.add_argument('--rotate_kv', type=str2bool, default=True)
    parser.add_argument('--group_rotate_kv', type=str2bool, default=False)
    parser.add_argument('--kv_quant_only', type=str2bool, default=False)
    parser.add_argument('--kv_tokenwise', type=str2bool, default=False)
    parser.add_argument('--gptq', type=str2bool, default=True)
    parser.add_argument('--gptq_percdamp', type=float, default=0.01)
    parser.add_argument('--gptq_cal_dataset', type=str, default='wikitext2',
                        choices=['wikitext2', 'ptb', 'c4'])
    parser.add_argument('--gptq_cal_nsamples', type=int, default=128)

    args = parser.parse_args()
    set_seed(args.seed)
    # local logging configuration
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    log_dir = f'./log'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir, f'{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}.txt')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.info(pprint.pformat(vars(args)))
    main(args)
