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
import datetime 
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
from accelerate import Accelerator
warnings.filterwarnings('ignore')

def main(args):
    #============================ Load model 
    if args.quarot: # QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs (arXiv:2404.00456)
        logging.info(f"Applying Hadamard transform following QuaRot")
        kwargs = {'device_map':'cpu','trust_remote_code':True,'attn_implementation':"eager"}
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype='auto' if args.auto_dtype else torch.float32, **kwargs)
        from scale_utils import rotation_utils
        rotation_utils.fuse_layer_norms(model)
        rotation_utils.rotate_model(model, args)
        rotation_utils.cleanup_memory(verbos=True)
    else:
        dev = 'cpu' if args.weight_prequant else 'balanced'
        kwargs = {'device_map':dev,'trust_remote_code':True,'attn_implementation':"eager"}
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype='auto' if args.auto_dtype else torch.float32, **kwargs)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model,trust_remote_code=True,use_fast=False)

    #============================ MX format 
    mx_specs_linear=parse_mx_specs(args,'linear')
    mx_specs_matmul=parse_mx_specs(args,'matmul')
    mx_specs_ln=parse_mx_specs(args,'ln')
    mx_specs_head=parse_mx_specs(args,'head')
    get_mx_model(
        model.eval(),
        mx_specs_linear=mx_specs_linear,
        mx_specs_matmul=mx_specs_matmul,
        mx_specs_ln=mx_specs_ln,
        mx_specs_head=mx_specs_head,
        args=args,
    )

    #============================ Runtime Hadamard Transform for QuaRot
    if args.quarot:
        from scale_utils import hadamard_utils
        if 'llama' in args.model or 'mistral' in args.model:
            for name, module in model.named_modules():
                if 'down_proj' in name:
                    had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
                    setattr(module, "online_full_had", True)
                    setattr(module, "had_K", had_K)
                    setattr(module, "K", K)
                if 'o_proj' in name:
                    had_K, K = hadamard_utils.get_hadK(model.config.num_attention_heads)
                    setattr(module, "online_partial_had", True)
                    setattr(module, "had_K", had_K)
                    setattr(module, "K", K)
                    setattr(module, "had_dim", model.config.hidden_size//model.config.num_attention_heads)
        else:
            raise NotImplementedError

        # KV Cache
        if args.rotate_kv:
            from scale_utils import model_utils, rotation_utils
            rope_function_name = model_utils.get_rope_function_name(model)
            layers = model_utils.get_layers(model)
            for layer in layers:
                rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                            layer.self_attn, 
                            rope_function_name, 
                            config=model.config,
                )

    # Load into GPU
    if model.device.type=='cpu':
        accelerator = Accelerator()
        model = accelerator.prepare(model)

    #============================ Evaluation 
    if args.eval_ppl:
        seqlen = 2048 # hard-coding
        args.limit = -1 # whole samples
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
            testloader = torch.load(cache_testloader)
            logging.info(f"load calibration from {cache_testloader}")
        else:
            from utils.calib import get_wikitext2_test
            testloader = get_wikitext2_test(seed=args.seed, seqlen=seqlen, model=args.model)
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
                if i==args.nsamples: break
                batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to('cuda')
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
                shift_labels = testenc[:, (i * seqlen) : ((i + 1) * seqlen)][
                    :, 1:
                ].to(model.lm_head.weight.device)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                neg_log_likelihood = loss.float() * seqlen
                nlls.append(neg_log_likelihood)
                pbar.set_description(f'loss: {loss.item():.4f}')
                if i == args.limit:
                    break
    
            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
        logging.info(f'wikitext ppl : {ppl.item()}')
        model.config.use_cache = use_cache
        results = {'wiki_ppl': ppl.item()}
    else: # lm-eval
        lm = lm_eval.models.huggingface.HFLM(
                pretrained=model,
                tokenizer=tokenizer,
                backend='causal',
                trust_remote_code=True,
            )
    
        with torch.no_grad():
            results = evaluator.simple_evaluate(
                model=lm,
                tasks=args.tasks,
                num_fewshot=args.num_fewshot,
                limit=args.nsamples,
            )
        results = results['results']
    logging.info(results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    # Model and Datsets
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--seed',type=int, default=0)
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
    parser.add_argument('--quarot', type=str2bool, default=False)
    parser.add_argument('--rotate_mode', type=str, default='hadamard')
    parser.add_argument('--rotate_kv', type=str2bool, default=True)
    parser.add_argument('--kv_quant_only', type=str2bool, default=False)
    parser.add_argument('--kv_tokenwise', type=str2bool, default=False)

    args = parser.parse_args()
    set_seed(args.seed)
    logging.info(args)
    main(args)
