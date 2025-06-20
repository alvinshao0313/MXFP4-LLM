import math
import time
import tqdm
import torch
import torch.nn as nn
import utils
import logging
from mx.elemwise_ops import quantize_elemwise_op
from mx.mx_ops import quantize_mx_op
from mx.specs import MxSpecs

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def mx_quant_dequant(input, mx_specs, axes=-1):
    dtype = input.dtype
    bf_in = quantize_elemwise_op(
        input.float(), mx_specs=mx_specs, round=mx_specs["round_output"]
    )
    qin_elem_format = mx_specs["w_elem_format"]
    qin = quantize_mx_op(
        bf_in,
        mx_specs,
        elem_format=qin_elem_format,
        scale_mode=mx_specs['W_scale_mode'],
        axes=[axes],
        round=mx_specs["round_mx_output"],
    )
    qin = qin.to(dtype)
    return qin


class WeightQuantizer(torch.nn.Module):
    '''From GPTQ Repo'''

    def __init__(self, shape=1):
        super(WeightQuantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(
        self, args, mtype
    ):
        self.mx_specs = MxSpecs()
        keys = ['scale_bits', 'block_size', 'w_elem_format', 'a_elem_format', 'A_elem_format', 'B_elem_format']
        for key in keys:
            try:
                val = vars(args)[f'{key}_{mtype}']
                self.mx_specs[key] = val if val != 'none' else None
            except:
                logging.info(f'[{mtype}] Set {key} to None')
                self.mx_specs[key] = None
                continue
        self.mx_specs['custom_cuda'] = args.custom_cuda
        self.mx_specs['a_scale_mode'] = args.a_scale_mode
        self.mx_specs['w_scale_mode'] = args.w_scale_mode
        self.mx_specs['A_scale_mode'] = args.A_scale_mode
        self.mx_specs['B_scale_mode'] = args.B_scale_mode
        self.mx_specs['per_tensor'] = args.per_tensor

    def find_params(self, x):
        if self.bits == 16:
            return
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            x = x.flatten(1)
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5)
            self.scale = xmax / self.maxq
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin.masked_fill(tmp, -1)
            xmax.masked_fill(tmp, +1)
            self.scale = (xmax - xmin).clamp(min=1e-5) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax

                if self.sym:
                    scale1 = xmax1 / self.maxq
                    zero1 = torch.zeros_like(scale1)
                    q = sym_quant_dequant(x, scale1.unsqueeze(1), self.maxq, self.fp4)
                else:
                    scale1 = (xmax1 - xmin1) / self.maxq
                    zero1 = torch.round(-xmin1 / scale1)
                    q = asym_quant_dequant(x, scale1.unsqueeze(1),
                                           zero1.unsqueeze(1), self.maxq, self.fp4)

                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]

        if not self.perchannel:
            tmp = shape[0]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        shape = [-1] + [1] * (len(shape) - 1)
        self.scale = self.scale.reshape(shape)
        self.zero = self.zero.reshape(shape)
        return

    # TODO: This should be better refactored into `forward`, which applies quantize and dequantize. A new method `quantize` should be added (if needed) to return the quantized integers and scales, like in ActQuantizer.
    def quantize(self, x):
        x_dtype = x.dtype
        if self.ready() and self.bits < 16:
            if self.sym:
                return sym_quant_dequant(x, self.scale, self.maxq, self.fp4).to(x_dtype)
            return asym_quant_dequant(x, self.scale, self.zero,
                                      self.maxq, self.fp4).to(x_dtype)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


class GPTQ:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):

        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False,
        static_groups=False, channel_protect=0,
    ):
        W = self.layer.weight.data.clone()
        if channel_protect > 0:
            protected_channels = self.layer.weight.data[:, :channel_protect].clone()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)])
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)])
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()

        if actorder:
            Q = Q[:, invperm]
        Q = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if channel_protect > 0:
            Q[:, :channel_protect] = protected_channels
        self.layer.weight.data = Q
        if torch.any(torch.isnan(self.layer.weight.data)):
            logging.warning('NaN in weights')
            import pprint
            pprint.pprint(self.quantizer.bits, self.quantizer.scale, self.quantizer.zero_point)
            raise ValueError('NaN in weights')

    def free(self):
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
        utils.cleanup_memory(verbos=False)


@torch.no_grad()
def gptq_fwrd(model, dataloader, dev, args):
    '''
    From GPTQ repo
    TODO: Make this function general to support both OPT and LLaMA models
    '''
    logging.info('-----GPTQ Quantization-----')
    print(dev)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    # for transformers >= 4.44.2,model.model has rotary emb
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    quantizers = {}
    llama_sequential = [
        ['self_attn.v_proj.module', 'self_attn.k_proj.module', 'self_attn.q_proj.module'],  #
        ['self_attn.o_proj.module'],
        ['mlp.up_proj.module', 'mlp.gate_proj.module'],
        ['mlp.down_proj.module'],
    ]
    deepseek_moe_sequential = [
        ['self_attn.k_proj.module', 'self_attn.v_proj.module', 'self_attn.q_proj.module'],
        ['self_attn.o_proj.module'],
        [*[f'mlp.experts.{i}.up_proj.module' for i in range(64)],
         *[f'mlp.experts.{i}.gate_proj.module' for i in range(64)],
         'mlp.shared_experts.up_proj.module', 'mlp.shared_experts.gate_proj.module'],
        [*[f'mlp.experts.{i}.down_proj.module' for i in range(64)],
         'mlp.shared_experts.down_proj.module']
    ]
    mixtral_sequential = [
        ['self_attn.k_proj.module', 'self_attn.v_proj.module', 'self_attn.q_proj.module'],
        ['self_attn.o_proj.module'],
        [f'block_sparse_moe.experts.{i}.w1.module' if j % 2 ==
            0 else f'block_sparse_moe.experts.{i}.w3.module' for i in range(8) for j in range(2)],
        [f'block_sparse_moe.experts.{i}.w2.module' for i in range(8)]
    ]
    for i in tqdm.tqdm(range(len(layers)), desc="(GPTQ Quant.) Layers"):
        logging.info(f'Layer {i}: ')
        layer = layers[i].to(dev)
        full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])
        if 'deepseek' in args.model.lower() and i >= 1:
            sequential = deepseek_moe_sequential
        elif 'mixtral' in args.model.lower():
            sequential = mixtral_sequential
        else:
            sequential = llama_sequential

        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                logging.info(f'{name}  ')
                layer_weight_bits = args.w_bits
                layer_weight_sym = not (args.w_asym)
                proj_keywords = ('k_proj', 'q_proj', 'v_proj', 'up_proj', 'gate_proj')
                if any(proj in name for proj in proj_keywords):
                    channel_protect = args.channel_protect
                else:
                    channel_protect = 0
                if 'lm_head' in name:
                    layer_weight_bits = 16
                    continue
                if args.w_bits_down_proj is not None and 'down_proj' in name:
                    layer_weight_bits = args.w_bits_down_proj
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = quant_utils.WeightQuantizer()
                gptq[name].quantizer.configure(
                    layer_weight_bits, perchannel=True, sym=layer_weight_sym,
                    mse=args.w_clip, fp4=args.w_fp4
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            for name in subset:
                layer_w_groupsize = args.w_groupsize
                gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=layer_w_groupsize,
                    actorder=args.act_order, static_groups=args.w_static_groups,
                    channel_protect=channel_protect
                )
                quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    utils.cleanup_memory(verbos=True)
    logging.info('-----GPTQ Quantization Done-----')
    return quantizers
