"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""

import torch
import numpy as np

from .mx_ops import quantize_mx_op, get_mx_quantize_params
from .elemwise_ops import quantize_elemwise_op
from .specs import apply_mx_specs, get_backwards_mx_specs
from .specs import mx_assert_test

torch_matmul = torch.matmul
torch_addmm = torch.addmm


class MatMulFunction(torch.autograd.Function):
    """Matches functionality of torch.matmul. Attempts to broadcast
    outmost dims if in1 and in2 have the same number of dims.
        in1: (..., out_rows, features)
        in2: (..., features, out_cols)
        out: (..., out_rows, out_cols)
    Otherwise, it expects the following shapes:
        in1: (..., out_rows, features)
        in2: (features, out_cols)
        out: (..., out_rows, out_cols)
    """

    @staticmethod
    def forward(ctx, in1, in2, bias, mx_specs, name, mode_config='aa', args=None, axes=[-1, -2]):
        dtype = in1.dtype
        assert mode_config in ["aa", "aw", "wa"]
        ctx.mode_config = mode_config
        if mode_config[0] == "a":
            qin1_elem_format = mx_specs["A_elem_format"]
        else:
            qin1_elem_format = mx_specs["w_elem_format"]

        if mode_config[1] == "a":
            qin2_elem_format = mx_specs["B_elem_format"]
        else:
            qin2_elem_format = mx_specs["w_elem_format"]

        bf_in1 = quantize_elemwise_op(
            in1.float(), mx_specs=mx_specs, round=mx_specs["round_output"]
        )
        bf_in2 = quantize_elemwise_op(
            in2.float(), mx_specs=mx_specs, round=mx_specs["round_output"]
        )

        if bias is not None:
            bf_bias = quantize_elemwise_op(
                bias.float(), mx_specs=mx_specs, round=mx_specs["round_weight"]
            ).to(dtype)

            ctx.bias_shape = list(bias.shape)
        else:
            bf_bias = None
            ctx.bias_shape = None

        if mx_specs["quantize_backprop"]:
            ctx.save_for_backward(bf_in1, bf_in2)
        else:
            ctx.save_for_backward(in1, in2)

        def scale_mode_to_elem_format(scale_mode):
            if scale_mode == 143:
                return 'fp8_e4m3'
            elif scale_mode == 152:
                return 'fp8_e5m2'
            elif scale_mode == 0:
                return 'e8m0'
            else:
                raise ValueError(f"Unsupported scale mode: {scale_mode}")
        # quantize along the dot product dimension
        if args.kv_quant_only:
            qin1 = bf_in1
        elif not mx_specs["double_quant"]:
            qin1 = quantize_mx_op(
                bf_in1,
                mx_specs,
                elem_format=qin1_elem_format,
                scale_mode=mx_specs['A_scale_mode'],
                axes=[axes[0]],
                round=mx_specs["round_mx_output"],
            )
        elif mx_specs["double_quant"]:
            assert 'asym' not in mx_specs['A_elem_format'], "Asymmetric quantization is not supported for double quantization"
            in1_scale, _, _, q_in1 = get_mx_quantize_params(
                bf_in1,
                mx_specs,
                elem_format=qin1_elem_format,
                scale_mode=2,
                axes=[axes[0]],
                round=mx_specs["round_mx_output"],
            )
            scale_mx_specs = mx_specs.copy()
            scale_mx_specs['per_tensor'] = True
            q_in1_scale = quantize_mx_op(
                in1_scale,
                scale_mx_specs,
                elem_format=scale_mode_to_elem_format(mx_specs['A_scale_mode']),
                scale_mode=2,
                axes=[axes[0]],
                round=mx_specs["round_mx_output"],
            )
            qin1 = q_in1_scale * q_in1
        if not mx_specs['double_quant']:
            qin2 = quantize_mx_op(
                bf_in2,
                mx_specs,
                elem_format=qin2_elem_format,
                scale_mode=mx_specs['B_scale_mode'],
                axes=[axes[1]],
                round=mx_specs["round_mx_output"],
            )
        elif mx_specs["double_quant"]:
            assert 'asym' not in mx_specs['B_elem_format'], "Asymmetric quantization is not supported for double quantization"
            in2_scale, _, _, q_in2 = get_mx_quantize_params(
                bf_in2,
                mx_specs,
                elem_format=qin2_elem_format,
                scale_mode=2,
                axes=[axes[1]],
                round=mx_specs["round_mx_output"],
            )
            scale_mx_specs = mx_specs.copy()
            scale_mx_specs['per_tensor'] = True
            q_in2_scale = quantize_mx_op(
                in2_scale,
                scale_mx_specs,
                elem_format=scale_mode_to_elem_format(mx_specs['B_scale_mode']),
                scale_mode=2,
                axes=[axes[1]],
                round=mx_specs["round_mx_output"],
            )
            qin2 = q_in2_scale * q_in2
        qin1 = qin1.to(dtype)
        qin2 = qin2.to(dtype)

        out = torch_matmul(qin1, qin2)
        out = quantize_elemwise_op(
            out, mx_specs=mx_specs, round=mx_specs["round_output"]
        )

        if bias is not None:
            out = out + bf_bias
            out = quantize_elemwise_op(
                out, mx_specs=mx_specs, round=mx_specs["round_output"]
            )

        ctx.mx_specs = get_backwards_mx_specs(mx_specs)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        """
        For a matmul in "wa" mode, the fwd and bwd matmuls configs are:
            FWD wt x act: w x a
            BWD wt x grad: w x a
            BWD act x grad: a x a <-- no mixed precision!
        """
        if ctx.mode_config[0] == "a":
            qin1_elem_format = ctx.mx_specs["a_elem_format_bp_os"]
        else:
            qin1_elem_format = ctx.mx_specs["w_elem_format_bp"]

        if ctx.mode_config[1] == "a":
            qin2_elem_format = ctx.mx_specs["a_elem_format_bp_os"]
        else:
            qin2_elem_format = ctx.mx_specs["w_elem_format_bp"]

        in1, in2 = ctx.saved_tensors

        grad_out = quantize_elemwise_op(
            grad_out,
            mx_specs=ctx.mx_specs,
            round=ctx.mx_specs["round_grad_input"],
        )

        #####################################################
        # perform madtile operation for grad_in1, grad_in2
        #####################################################
        qin1 = quantize_mx_op(
            in1,
            ctx.mx_specs,
            elem_format=qin1_elem_format,
            axes=[-2],
            round=ctx.mx_specs["round_mx_input_grad_input"],
        )
        qin2 = quantize_mx_op(
            in2,
            ctx.mx_specs,
            elem_format=qin2_elem_format,
            axes=[-1],
            round=ctx.mx_specs["round_mx_input_grad_input"],
        )

        # quantize along out_cols
        qgrad_out1 = quantize_mx_op(
            grad_out,
            ctx.mx_specs,
            elem_format=ctx.mx_specs["a_elem_format_bp_os"],
            axes=[-1],
            round=ctx.mx_specs["round_mx_grad_output_grad_input"],
        )
        # quantize along out_rows
        qgrad_out2 = quantize_mx_op(
            grad_out,
            ctx.mx_specs,
            elem_format=ctx.mx_specs["a_elem_format_bp_os"],
            axes=[-2],
            round=ctx.mx_specs["round_mx_grad_output_grad_input"],
        )

        # compute grad_in1 and grad_in2
        grad_in1 = torch_matmul(qgrad_out1, qin2.transpose(-1, -2))
        grad_in2 = torch_matmul(qin1.transpose(-1, -2), qgrad_out2)

        # element-wise quantize for grad_in1 and grad_in2
        grad_in1 = quantize_elemwise_op(
            grad_in1,
            mx_specs=ctx.mx_specs,
            round=ctx.mx_specs["round_grad_input"],
        )
        grad_in2 = quantize_elemwise_op(
            grad_in2,
            mx_specs=ctx.mx_specs,
            round=ctx.mx_specs["round_grad_input"],
        )

        #####################################################
        # Compute grad_bias
        #####################################################
        if ctx.bias_shape is None:
            grad_bias = None
        else:
            inner_size = grad_out.shape[-1]
            assert np.prod(ctx.bias_shape) == inner_size
            grad_bias = grad_out.reshape(-1, inner_size).sum(0)
            grad_bias = grad_bias.reshape(ctx.bias_shape)

            grad_bias = quantize_elemwise_op(
                grad_bias,
                mx_specs=ctx.mx_specs,
                round=ctx.mx_specs["round_grad_weight"],
            )

        return (grad_in1, grad_in2, grad_bias, None, None, None)


def matmul(in1, in2, bias=None, mx_specs=None, name=None, mode_config='aa', args=None, axes=[-1, -2]):
    mx_assert_test(mx_specs)
    if mx_specs is None:
        if bias is None:
            out = torch_matmul(in1, in2)
        else:
            out = torch_addmm(bias, in1, in2)
        return out

    mx_specs = apply_mx_specs(mx_specs)

    return MatMulFunction.apply(in1, in2, bias, mx_specs, name, mode_config, args, axes)
