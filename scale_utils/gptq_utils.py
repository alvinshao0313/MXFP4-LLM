import math
import time
import tqdm
import torch
import torch.nn as nn
import utils
import logging
from mx.elemwise_ops import quantize_elemwise_op
from mx.mx_ops import get_mx_quantize_params, apply_mx_quantize_with_param
from mx.specs import MxSpecs
# from mx import Linear
try:
    from scale_utils import hadamard_utils
    import fast_hadamard_transform
except:
    print('hadamard_utils is not imported')

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def find_qlayers(module, layers=[torch.nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_qlayers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


class WeightQuantizer(torch.nn.Module):
    def __init__(self):
        super(WeightQuantizer, self).__init__()
        self.mx_specs = MxSpecs()
        self.scale1 = torch.zeros(1, dtype=torch.float32)
        self.scale2 = torch.zeros(1, dtype=torch.float32)
        self.shift = torch.zeros(1, dtype=torch.float32)

    def configure(self, args, mtype='linear'):
        assert not args.per_tensor, "GPTQ does not support per_tensor quantization"
        keys = ['scale_bits', 'block_size', 'w_elem_format']
        for key in keys:
            try:
                val = vars(args)[f'{key}_{mtype}']
                self.mx_specs[key] = val if val != 'none' else None
            except:
                logging.info(f'[{mtype}] Set {key} to None')
                self.mx_specs[key] = None
                continue
        self.mx_specs['custom_cuda'] = args.custom_cuda
        self.mx_specs['w_scale_mode'] = args.w_scale_mode

    def forward(self, x, block_size=1):
        dtype = x.dtype
        bf_x = quantize_elemwise_op(
            x.float(), mx_specs=self.mx_specs, round=self.mx_specs["round_output"]
        )
        qx = apply_mx_quantize_with_param(
            bf_x,
            self.scale1.float(),
            self.scale2.float(),
            self.shift.float(),
            self.mx_specs,
            block_size=block_size,
            elem_format=self.mx_specs["w_elem_format"],
            scale_mode=self.mx_specs['w_scale_mode'],
            axes=[-1],
            round=self.mx_specs["round_mx_output"],
        )
        qx = qx.to(dtype)
        return qx

    def find_params(self, x):
        bf_in = quantize_elemwise_op(
            x.float(), mx_specs=self.mx_specs, round=self.mx_specs["round_output"]
        )
        self.scale1, self.scale2, self.shift, _ = get_mx_quantize_params(
            bf_in,
            self.mx_specs,
            elem_format=self.mx_specs['w_elem_format'],
            scale_mode=self.mx_specs['w_scale_mode'],
            axes=[-1],
            round=self.mx_specs["round_mx_output"],
        )
        self.scale1 = self.scale1[:, 0]
        self.scale2 = self.scale2[:, 0]
        self.shift = self.shift[:, 0]
        return

    def ready(self):
        return torch.all(self.scale1 != 0)


class GPTQ:
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.dtype = self.layer.weight.dtype
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
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self, blocksize=128, groupsize=-1,
        percdamp=.01, actorder=False
    ):
        W = self.layer.weight.data.clone()
        W = W.float()
        if not self.quantizer.ready():
            self.quantizer.find_params(W)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

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
                    if (i1 + i) % groupsize == 0:
                        self.quantizer.find_params(
                            W[:, (i1 + i):(i1 + i + groupsize)])

                q = self.quantizer(w.unsqueeze(0)).flatten()
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
        self.layer.weight.data = Q.reshape(
            self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if torch.any(torch.isnan(self.layer.weight.data)):
            logging.warning('NaN in weights')
            raise ValueError('NaN in weights')

    def free(self):
        self.H = None
        torch.cuda.empty_cache()


def rotate_pre_hook(rotate_mode):
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


def register_rotate_hook(model, args):
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
                    rotate_pre_hook(rotate_mode)))
            elif 'o_proj' in name and args.online_partial_had:
                had_K, K = hadamard_utils.get_hadK(
                    model.config.num_attention_heads)
                rotate_mode["online_partial_had"] = True
                rotate_mode["had_K"] = had_K
                rotate_mode["K"] = K
                rotate_mode["had_dim"] = model.config.hidden_size // model.config.num_attention_heads
                rotate_handles.append(mod.register_forward_pre_hook(
                    rotate_pre_hook(rotate_mode)))
    else:
        raise NotImplementedError
    return rotate_handles


@torch.no_grad()
def gptq_fwrd(model, dataloader, dev, args):
    '''
    From GPTQ repo
    TODO: Make this function general to support both OPT and LLaMA models
    '''
    logging.info('-----GPTQ Quantization-----')
    if args.rotate:
        rotate_handles = register_rotate_hook(model, args)
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    if hasattr(model.model, "rotary_emb"):  # transformers >= 4.44.2,model.model has rotary emb
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.gptq_cal_nsamples, args.gptq_cal_seqlen, model.config.hidden_size), dtype=dtype, device=dev
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

    llama_sequential = [
        ['self_attn.v_proj', 'self_attn.k_proj', 'self_attn.q_proj'],  #
        ['self_attn.o_proj'],
        ['mlp.up_proj', 'mlp.gate_proj'],
        ['mlp.down_proj'],
    ]
    deepseek_moe_sequential = [
        ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
        ['self_attn.o_proj'],
        [*[f'mlp.experts.{i}.up_proj' for i in range(64)],
         *[f'mlp.experts.{i}.gate_proj' for i in range(64)],
         'mlp.shared_experts.up_proj', 'mlp.shared_experts.gate_proj'],
        [*[f'mlp.experts.{i}.down_proj' for i in range(64)],
         'mlp.shared_experts.down_proj']
    ]
    mixtral_sequential = [
        ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
        ['self_attn.o_proj'],
        [f'block_sparse_moe.experts.{i}.w1' if j % 2 ==
            0 else f'block_sparse_moe.experts.{i}.w3' for i in range(8) for j in range(2)],
        [f'block_sparse_moe.experts.{i}.w2' for i in range(8)]
    ]
    for i in tqdm.tqdm(range(len(layers)), desc="(GPTQ Quant.) Layers"):
        layer = layers[i].to(dev)
        full = find_qlayers(layer, layers=[torch.nn.Linear])
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
                if 'lm_head' in name:
                    continue
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = WeightQuantizer()
                gptq[name].quantizer.configure(args, mtype='linear')

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(
                    subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.gptq_cal_nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            for name in subset:
                gptq[name].fasterquant(
                    percdamp=args.gptq_percdamp, groupsize=args.block_size_linear)
                gptq[name].free()

        for j in range(args.gptq_cal_nsamples):
            outs[j] = layer(inps[j].unsqueeze(
                0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if args.rotate:
        for h in rotate_handles:
            h.remove()
    model.config.use_cache = use_cache
    logging.info('-----GPTQ Quantization Done-----')
