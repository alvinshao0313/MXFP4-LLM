"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""

import torch
import torch.nn.functional as F
import torch.nn as nn

from .mx_ops import quantize_mx_op, get_mx_quantize_params
from .elemwise_ops import quantize_elemwise_op
from .specs import apply_mx_specs, get_backwards_mx_specs
from .specs import mx_assert_test

try:
    from scale_utils import hadamard_utils
    import fast_hadamard_transform
except:
    print('hadamard_utils is not imported')
import math

f_linear = F.linear
torch_matmul = torch.matmul


class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        bias=None,
        mx_specs=None,
        name=None,
        args=None,
    ):
        dtype = input.dtype
        # element-wise quantize for input
        bf_in = quantize_elemwise_op(
            input.float(), mx_specs=mx_specs, round=mx_specs["round_output"]
        )

        # element-wise quantize for weight and bias
        bf_weight = quantize_elemwise_op(
            weight.float(), mx_specs=mx_specs, round=mx_specs["round_weight"]
        )

        if bias is not None:
            ctx.has_bias = True
            bf_bias = quantize_elemwise_op(
                bias.float(), mx_specs=mx_specs, round=mx_specs["round_weight"]
            ).to(dtype)
        else:
            ctx.has_bias = False
            bf_bias = None

        if mx_specs["quantize_backprop"]:
            ctx.save_for_backward(bf_in, bf_weight)
        else:
            ctx.save_for_backward(input, weight)

        # MX quantize everything along input size
        if not mx_specs['double_quant']:
            qis_input = quantize_mx_op(
                bf_in,
                mx_specs,
                elem_format=mx_specs['a_elem_format'],
                scale_mode=mx_specs['a_scale_mode'],
                axes=[-1],
                round=mx_specs["round_mx_output"],
            )
            qis_weight = quantize_mx_op(
                bf_weight,
                mx_specs,
                elem_format=mx_specs['w_elem_format'],
                scale_mode=mx_specs['w_scale_mode'],
                axes=[-1],
                round=mx_specs["round_mx_output"],
            )
        elif mx_specs['double_quant']:  # double quantization
            if mx_specs['a_elem_format']:
                assert 'asym' not in mx_specs['a_elem_format'], "Asymmetric quantization is not supported for double quantization"
            assert 'asym' not in mx_specs['w_elem_format'], "Asymmetric quantization is not supported for double quantization"
            assert mx_specs['a_scale_mode'] in [
                143, 152, 0], "Only scale_mode 143, 152, or 0 is supported for double quantization"
            assert mx_specs['w_scale_mode'] in [
                143, 152, 0], "Only scale_mode 143, 152, or 0 is supported for double quantization"

            def scale_mode_to_elem_format(scale_mode):
                if scale_mode == 143:
                    return 'fp8_e4m3'
                elif scale_mode == 152:
                    return 'fp8_e5m2'
                elif scale_mode == 0:
                    return 'e8m0'
                else:
                    raise ValueError(f"Unsupported scale mode: {scale_mode}")
            w_scale, _, _, q_w = get_mx_quantize_params(
                bf_weight,
                mx_specs,
                elem_format=mx_specs['w_elem_format'],
                scale_mode=2,
                axes=[-1],
                round=mx_specs["round_mx_output"],
            )
            scale_mx_specs = mx_specs.copy()
            if args.w_dq_only:
                qis_input = quantize_mx_op(
                    bf_in,
                    mx_specs,
                    elem_format=mx_specs['a_elem_format'],
                    scale_mode=mx_specs['a_scale_mode'],
                    axes=[-1],
                    round=mx_specs["round_mx_output"],
                )
                scale_mx_specs['block_size'] = -1
            else:
                in_scale, _, _, q_in = get_mx_quantize_params(
                    bf_in,
                    mx_specs,
                    elem_format=mx_specs['a_elem_format'],
                    scale_mode=2,
                    axes=[-1],
                    round=mx_specs["round_mx_output"],
                )
                scale_mx_specs['per_tensor'] = True
                q_in_scale = quantize_mx_op(
                    in_scale,
                    scale_mx_specs,
                    elem_format=scale_mode_to_elem_format(mx_specs['a_scale_mode']),
                    scale_mode=2,
                    axes=[-1],
                    round=scale_mx_specs["round_mx_output"],
                )
                qis_input = q_in_scale * q_in

            q_w_scale = quantize_mx_op(
                w_scale,
                scale_mx_specs,
                elem_format=scale_mode_to_elem_format(mx_specs['w_scale_mode']),
                scale_mode=2,
                axes=[-1],
                round=scale_mx_specs["round_mx_output"],
            )
            qis_weight = q_w_scale * q_w

        qis_input = qis_input.to(dtype)
        qis_weight = qis_weight.to(dtype)

        # compute output
        output = f_linear(qis_input, qis_weight)

        output = quantize_elemwise_op(
            output, mx_specs=mx_specs, round=mx_specs["round_output"]
        )

        if bias is not None:
            output = output + bf_bias
            output = quantize_elemwise_op(
                output, mx_specs=mx_specs, round=mx_specs["round_output"]
            )

        ctx.mx_specs = get_backwards_mx_specs(mx_specs)
        ctx.name = name
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # load context
        input, weight = ctx.saved_tensors

        out_dim = weight.shape[0]
        in_dim = weight.shape[1]

        grad_output = quantize_elemwise_op(
            grad_output,
            mx_specs=ctx.mx_specs,
            round=ctx.mx_specs["round_grad_input"],
        )

        #####################################################
        # perform madtile operation for grad_weight, grad_bias
        #####################################################
        # if the input is 2D, quantize everything along examples (batches)
        # if the input is 3D, quantize everything along the first axis
        qex_input = quantize_mx_op(
            input,
            ctx.mx_specs,
            elem_format=ctx.mx_specs['a_elem_format_bp_ex'],
            axes=[-2],
            round=ctx.mx_specs["round_mx_input_grad_weight"],
        )
        qex_grad_output = quantize_mx_op(
            grad_output,
            ctx.mx_specs,
            elem_format=ctx.mx_specs['a_elem_format_bp_ex'],
            axes=[-2],
            round=ctx.mx_specs["round_mx_grad_output_grad_weight"],
        )

        # compute grad_weight [out_features, in_features]
        qex_grad_output = qex_grad_output.reshape(-1, out_dim)
        qex_input = qex_input.reshape(-1, in_dim)

        # Compute grad_weight
        grad_weight = torch_matmul(qex_grad_output.transpose(0, 1), qex_input)
        grad_weight = quantize_elemwise_op(
            grad_weight,
            mx_specs=ctx.mx_specs,
            round=ctx.mx_specs["round_grad_weight"],
        )

        #####################################################
        # perform madtile operation for grad_input
        #####################################################
        # compute grad_input, quantize everything along output size
        qos_weight = quantize_mx_op(
            weight,
            ctx.mx_specs,
            elem_format=ctx.mx_specs['w_elem_format_bp'],
            axes=[0],
            round=ctx.mx_specs["round_mx_weight_grad_input"],
        )
        # grad_output shape is (B, seq, out_dim)
        qos_grad_output = quantize_mx_op(
            grad_output,
            ctx.mx_specs,
            elem_format=ctx.mx_specs['a_elem_format_bp_os'],
            axes=[-1],
            round=ctx.mx_specs["round_mx_grad_output_grad_input"],
        )

        # Compute grad_input
        grad_input = torch_matmul(qos_grad_output, qos_weight)
        grad_input = quantize_elemwise_op(
            grad_input,
            mx_specs=ctx.mx_specs,
            round=ctx.mx_specs["round_grad_input"],
        )

        #####################################################
        # Compute grad_bias
        #####################################################
        if not ctx.has_bias:
            grad_bias = None
        else:
            grad_bias = grad_output.reshape(-1, out_dim).sum(0)
            grad_bias = quantize_elemwise_op(
                grad_bias,
                mx_specs=ctx.mx_specs,
                round=ctx.mx_specs["round_grad_weight"],
            )

        return (grad_input, grad_weight, grad_bias, None, None, None, None)


def linear(
    input,
    weight,
    bias=None,
    mx_specs=None,
    name=None,
    args=None,
):
    mx_assert_test(mx_specs)
    if mx_specs is None:
        return f_linear(input, weight, bias=bias)

    mx_specs = apply_mx_specs(mx_specs)

    return LinearFunction.apply(input, weight, bias, mx_specs, name, args)


class Linear(torch.nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        mx_specs=None,
        name=None,
        args=None,
    ):
        mx_assert_test(mx_specs)
        self.mx_none = mx_specs is None

        self.name = name
        self.mx_specs = apply_mx_specs(mx_specs)
        super().__init__(in_features, out_features, bias)
        self.args = args

    def apply_mx_specs(self, mx_specs):
        mx_assert_test(mx_specs)
        self.mx_none = mx_specs is None
        self.mx_specs = apply_mx_specs(mx_specs)

    def append_name(self, postfix):
        self.name += postfix

    def forward(self, inputs):
        if self.mx_none:
            return super().forward(inputs)

        # Hadamard transform (QuaRot)
        if hasattr(self, "online_full_had"):
            inputs = hadamard_utils.matmul_hadU_cuda(inputs, self.had_K, self.K)
        elif hasattr(self, "online_partial_had"):
            init_shape = inputs.shape
            if self.K == 1:
                inputs = fast_hadamard_transform.hadamard_transform(
                    inputs.reshape(-1, init_shape[-1] // self.had_dim, self.had_dim).transpose(1, 2),
                    scale=1 / math.sqrt(init_shape[-1] // self.had_dim)
                ).transpose(1, 2)
            else:
                inputs = (self.had_K.to(inputs.dtype).to(inputs.device) @ inputs.reshape(-1,
                          init_shape[-1] // self.had_dim, self.had_dim)) / math.sqrt(init_shape[-1] // self.had_dim)
            inputs = inputs.reshape(init_shape)
        elif hasattr(self, "online_group_had"):
            assert self.had_dim > 0 and self.K == 1, "Group Hadamard transform requires had_dim > 0 and K == 1"
            # Group Hadamard transform
            init_shape = inputs.shape
            inputs = fast_hadamard_transform.hadamard_transform(
                inputs.reshape(-1, init_shape[-1] // self.had_dim, self.had_dim),
                scale=1 / math.sqrt(self.had_dim))
            inputs = inputs.reshape(init_shape)

        out = linear(
            input=inputs,
            weight=self.weight,
            bias=self.bias,
            mx_specs=self.mx_specs,
            name=self.name,
            args=self.args,
        )
        return out
