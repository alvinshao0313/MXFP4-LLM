import os
from datasets import load_dataset
from transformers import AutoTokenizer 
import torch
import random

def get_c4(nsamples, seed, seqlen, model):
    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, trust_remote_code=True)

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader

def get_wikitext2(nsamples, seed, seqlen, model):
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')#, revision="script")

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, trust_remote_code=True)
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader

def get_wikitext2_test(seed, seqlen, model):
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, trust_remote_code=True)
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    return testenc

def get_loaders(name, nsamples=128, seed=0, seqlen=2048, model=''):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model)
    elif 'c4' in name:
        return get_c4(nsamples, seed, seqlen, model)
    else:
        raise NotImplementedError

def get_calibset(args, nsamples=None, seqlen=None, calibset='wikitext2'):
    # set nsamples and seqlen
    if nsamples is None:
        nsamples = args.nsamples
    else:
        print(f'[utils/calib.py:get_calibset] You force the nsamples to {nsamples}')
    if seqlen is None:
        seqlen = model.seqlen
    else:
        print(f'[utils/calib.py:get_calibset] You force the model.seqlen to {seqlen}')
    # Data cache
    if 'llama3' in args.model:
        dataset_cache = f'calibset/{calibset}_{nsamples}_{seqlen}_{args.seed}_llama3.cache'
    elif 'llama2' in args.model:
        dataset_cache = f'calibset/{calibset}_{nsamples}_{seqlen}_{args.seed}_llama2.cache'
    elif 'llama' in args.model:
        dataset_cache = f'calibset/{calibset}_{nsamples}_{seqlen}_{args.seed}_llama.cache'
    elif 'mistral' in args.model:
        dataset_cache = f'calibset/{calibset}_{nsamples}_{seqlen}_{args.seed}_mistral.cache'
    elif 'qwen2' in args.model:
        dataset_cache = f'calibset/{calibset}_{nsamples}_{seqlen}_{args.seed}_qwen2.cache'
    elif 'qwen' in args.model:
        dataset_cache = f'calibset/{calibset}_{nsamples}_{seqlen}_{args.seed}_qwen.cache'
    elif 'gpt' in args.model:
        dataset_cache = f'calibset/{calibset}_{nsamples}_{seqlen}_{args.seed}_gpt.cache'
    elif 'opt' in args.model:
        dataset_cache = f'calibset/{calibset}_{nsamples}_{seqlen}_{args.seed}_opt.cache'
    elif 'midm' in args.model:
        dataset_cache = f'calibset/{calibset}_{nsamples}_{seqlen}_{args.seed}_midm.cache'
    else:
        dataset_cache = f'calibset/{calibset}_{nsamples}_{seqlen}_{args.seed}.cache'
    # Get calibset
    try:
        dataloader = torch.load(dataset_cache)
        print(f'Load dataloader from {dataset_cache}')
    except:
        dataloader = get_loaders(
            calibset, nsamples=nsamples, seed=args.seed, model=args.model, seqlen=seqlen
        )
        if not os.path.exists('calibset'):
            os.mkdir('calibset')
        torch.save(dataloader, dataset_cache)
    return dataloader
