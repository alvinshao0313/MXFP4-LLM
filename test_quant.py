# %%
from mx.elemwise_ops import quantize_elemwise_op
from mx.mx_ops import quantize_mx_op, apply_mx_quantize_with_param, get_mx_quantize_params
from mx.specs import MxSpecs

# %%


def mx_quant_dequant(input, mx_specs, axes=-1):
    dtype = input.dtype
    bf_in = quantize_elemwise_op(
        input.float(), mx_specs=mx_specs, round=mx_specs["round_output"]
    )
    qin = quantize_mx_op(
        bf_in,
        mx_specs,
        elem_format=mx_specs['w_elem_format'],
        scale_mode=mx_specs['w_scale_mode'],
        axes=[axes],
        round=mx_specs["round_mx_output"],
    )
    qin = qin.to(dtype)
    return qin


def get_quant_params(input, mx_specs, axes=-1):
    bf_in = quantize_elemwise_op(
        input.float(), mx_specs=mx_specs, round=mx_specs["round_output"]
    )
    scale1, scale2, shift = get_mx_quantize_params(
        bf_in,
        mx_specs,
        elem_format=mx_specs['w_elem_format'],
        scale_mode=mx_specs['w_scale_mode'],
        axes=[axes],
        round=mx_specs["round_mx_output"],
    )
    return scale1, scale2, shift


def apply_quant_with_params(input, scale1, scale2, shift, mx_specs, axes=-1):
    dtype = input.dtype
    bf_in = quantize_elemwise_op(
        input.float(), mx_specs=mx_specs, round=mx_specs["round_output"]
    )
    qin = apply_mx_quantize_with_param(
        bf_in,
        scale1.float(),
        scale2.float(),
        shift.float(),
        mx_specs=mx_specs,
        elem_format=mx_specs['w_elem_format'],
        scale_mode=mx_specs['w_scale_mode'],
        axes=[axes],
        round=mx_specs["round_mx_output"],
    )
    qin = qin.to(dtype)
    return qin


# %%
mx_specs = MxSpecs()
mx_specs['custom_cuda'] = True
mx_specs['w_elem_format'] = 'int4'
mx_specs['w_scale_mode'] = 2
mx_specs['per_tensor'] = False
mx_specs['scale_bits'] = 16
mx_specs['block_size'] = -1

# %%
import torch
tensor = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 400, -1]], dtype=torch.float32).to('cuda')
quantized_tensor = mx_quant_dequant(tensor, mx_specs, axes=-1)
print(quantized_tensor)

scale1, scale2, shift = get_quant_params(
    tensor, mx_specs=mx_specs, axes=-1,)
print(f"Scale1: {scale1}, Scale2: {scale2}, Shift: {shift}")
mx_specs['block_size'] = 1
q_t = apply_quant_with_params(
    tensor[:, 0], scale1, scale2, shift,
    mx_specs=mx_specs, axes=-1,)
print(q_t)

# %%
tensor = tensor.reshape(3, 4)
q_t = (tensor / (tensor.max(dim=-1, keepdim=True)[0] / 7)
       ).round().clamp(-8, 7) * (tensor.max(dim=-1, keepdim=True)[0] / 7)
print(q_t)

# %%
for k, v in mx_specs.items():
    print(f"{k}: {v}")
