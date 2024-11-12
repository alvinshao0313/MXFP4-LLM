"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""

import pytest
import torch
import numpy as np
import sys

from .common_lib import (
        check_diff,
        check_diff_quantize,
        all_encodings
)

from mx.formats import _get_format_params
from mx.mx_ops import _quantize_mx

np.random.seed(0xd10)

DEVICE__CUSTOM_CUDA = [
#    ("cpu",  False),
#    ("cpu",  True),
#    ("cuda", True),
    ("cuda", False),
]

ELEM_FMTS = [
    ("fp8_e5m2"),
    ("fp8_e4m3"),
    ("fp6_e3m2"),
    ("fp6_e2m3"),
    ("fp4_e2m1"),
    ("int4"),
]


@pytest.mark.parametrize("scale_bits", (8,7,6,5,4,3,2))
@pytest.mark.parametrize("elem_format", ELEM_FMTS)
#@pytest.mark.parametrize("block_size", (8, 9, 64))
@pytest.mark.parametrize("block_size", (32,))
#@pytest.mark.parametrize("round", ('nearest', 'floor', 'even'))
@pytest.mark.parametrize("round", ('nearest',))
#@pytest.mark.parametrize("flush_fp32_subnorms", (False,True))
#@pytest.mark.parametrize("flush_fp32_subnorms", (True,))
@pytest.mark.parametrize("flush_fp32_subnorms", (False,))
@pytest.mark.parametrize("device, custom_cuda", DEVICE__CUSTOM_CUDA)
def test_mx_encoding(scale_bits, elem_format, block_size, round,
                     flush_fp32_subnorms, device, custom_cuda):

    x1 = all_encodings(8, 9, device="cpu")
    x1 = x1.half()
    #x1 = x1.bfloat16()
    x2 = x1.clone().detach().to(device)

    # FP32
    y1 = _quantize_mx(x1.float(), scale_bits, elem_format,
                      block_size=block_size,
                      axes=[-1],
                      round=round,
                      flush_fp32_subnorms=flush_fp32_subnorms,
                      custom_cuda=False)

    # FP16
    y2 = _quantize_mx(x2, scale_bits, elem_format,
                      block_size=block_size,
                      axes=[-1],
                      round=round,
                      flush_fp32_subnorms=flush_fp32_subnorms,
                      custom_cuda=False)

    check_diff_quantize(x1, y1, y2)
