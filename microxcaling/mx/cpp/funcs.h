/*
 * Microsoft Confidential
 */

#ifndef PYT_MX_FUNCS_H
#define PYT_MX_FUNCS_H

#include <torch/extension.h>
#include "common.cuh"

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

//-----------------------------------------------------------------------
// CUDA forward declarations
//-----------------------------------------------------------------------
torch::Tensor quantize_mx_cuda(
    const torch::Tensor A, // Input tensor
    const int scale_bits,
    int elem_ebits,
    int elem_mbits,
    float elem_max_norm,
    const torch::Tensor max_values,
    const torch::Tensor pos_values,
    const torch::Tensor neg_values,
    const torch::Tensor std_values,
    const int axis,
    const bool flush_fp32_subnorms = false,
    const RoundingMode rounding_mode = rd_away,
    const int scale_mode = 0,
    const int asym = -1
);

torch::Tensor quantize_mx_by_tile_cuda(
    const torch::Tensor input,
    const int scale_bits,
    int elem_ebits,
    int elem_mbits,
    float elem_max_norm,
    const int tile_size,
    const int axis,
    const bool flush_fp32_subnorms = false,
    const RoundingMode rounding_mode = rd_away,
    const int scale_mode = 0,
    const int asym = -1
);

torch::Tensor quantize_elemwise_cuda(
    const torch::Tensor A, // Input tensor
    const int bits,     // Number of bits (sign + magnitude) to quantize to
    const int exp_bits, // Number of exponent bits to quantize to
    const float max_norm,
    const RoundingMode rounding_mode = rd_away,
    const bool saturate_normals = false,
    const bool allow_denorm = true
);

torch::Tensor reduce_sum_inner_dim(
    const torch::Tensor A
);

torch::Tensor reduce_max_inner_dim(
    const torch::Tensor A
);

std::vector<torch::Tensor> get_mx_quantize_param_by_tile_func_cuda(
    torch::Tensor A,
    int scale_bits,
    int elem_ebits,
    int elem_mbits,
    float elem_max_norm,
    int tile_size,
    int axis,
    const bool flush_fp32_subnorms = false,
    const int rmode = 0,
    const int scale_mode = 0,
    const int asym = -1
);

std::vector<torch::Tensor> get_mx_quantize_param_by_tile_cuda(
    const torch::Tensor input,
    const int scale_bits,
    int elem_ebits,
    int elem_mbits,
    float elem_max_norm,
    const int tile_size,
    const int axis,
    const bool flush_fp32_subnorms = false,
    const RoundingMode rounding_mode = rd_away,
    const int scale_mode = 0,
    const int asym = -1
);

torch::Tensor apply_mx_quantize_with_param_func_cuda(
    torch::Tensor A,
    torch::Tensor scales1,
    torch::Tensor scales2,
    torch::Tensor shifts,
    int elem_ebits,
    int elem_mbits,
    float elem_max_norm,
    int tile_size,
    int axis,
    const bool flush_fp32_subnorms = false,
    const int rmode = 0,
    const int scale_mode = 0,
    const int asym = -1
);

torch::Tensor apply_mx_quantize_with_param_cuda(
    const torch::Tensor input,
    const torch::Tensor scales1,
    const torch::Tensor scales2,
    const torch::Tensor shifts,
    int elem_ebits,
    int elem_mbits,
    float elem_max_norm,
    const int tile_size,
    const int axis,
    const bool flush_fp32_subnorms = false,
    const RoundingMode rounding_mode = rd_away,
    const int scale_mode = 0,
    const int asym = -1
);


#endif
