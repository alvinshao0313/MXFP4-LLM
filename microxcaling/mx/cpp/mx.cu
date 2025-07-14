#include <torch/types.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAUtils.h>
#include <c10/cuda/CUDAGuard.h>
#include "common.cuh"
#include "mx.cuh"

//-----------------------------------------------------------------------
// quantize_mx_cuda
//-----------------------------------------------------------------------
torch::Tensor quantize_mx_cuda(
    const torch::Tensor input,
    const int scale_bits,
    int elem_ebits,
    int elem_mbits,
    float elem_max_norm,
    const torch::Tensor max_values, // absmax values
    const torch::Tensor pos_values, // positive values
    const torch::Tensor neg_values, // negative values
    const torch::Tensor std_values, // std values
    const int axis,
    const bool flush_fp32_subnorms = false,
    const RoundingMode rounding_mode = rd_away,
    const int scale_mode = 0,
    const int asym = -1
) {
    at::Device device = input.device();
    const at::cuda::CUDAGuard device_guard{device};
    auto output = torch::empty_like(input);
    output = output.to(device);

    const int ndim = input.dim();
    auto input_sizes = input.sizes();

    // Size of shared axis
    const int axis_size = input_sizes[axis];
    // Size of axes before shared axis
    long pre_axis_size = 1;
    for (int i = 0; i < axis; i++) {
        pre_axis_size *= input_sizes[i];
    }
    // Size of axes after shared axis
    long post_axis_size = 1;
    for (int i = axis + 1; i < ndim; i++) {
        post_axis_size *= input_sizes[i];
    }

    long total_size = pre_axis_size * axis_size * post_axis_size;
    // 1 thread per element, up to max number of threads
    const long blocks = get_blocks(total_size);
    const int threads = get_threads(total_size);

    // Call CUDA kernel
    if (input.dtype() == torch::ScalarType::Half) {
        AT_ASSERTM(0, " fp16 not supported for MX");
    } else {
        quantize_mx_cuda_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(),
            scale_bits,
            elem_ebits,
            elem_mbits,
            elem_max_norm,
            max_values.data_ptr<float>(),
            pos_values.data_ptr<float>(),
            neg_values.data_ptr<float>(),
            std_values.data_ptr<float>(),
            total_size,
            axis_size,
            post_axis_size,
            flush_fp32_subnorms,
            rounding_mode,
            scale_mode,
            asym,
            output.data_ptr<float>()
        );
    }

    gpuErrchk(cudaPeekAtLastError());
    return output;
}

//-----------------------------------------------------------------------
// quantize_mx_by_tile
//-----------------------------------------------------------------------
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
) {
    at::Device device = input.device();
    const at::cuda::CUDAGuard device_guard{device};
    auto output = torch::empty_like(input);
    output = output.to(device);

    const int ndim = input.dim();
    auto input_sizes = input.sizes();

    // Size of shared axis
    const int axis_size = input_sizes[axis];
    int tsize = (tile_size > 0) ? tile_size : axis_size;
    // Size of axes before shared axis
    long pre_axis_size = 1;
    for (int i = 0; i < axis; i++) {
        pre_axis_size *= input_sizes[i];
    }
    // Size of axes after shared axis
    long post_axis_size = 1;
    for (int i = axis + 1; i < ndim; i++) {
        post_axis_size *= input_sizes[i];
    }
    // Number of tiles along the chosen axis
    int num_tiles = axis_size / tsize;
    if (axis_size % tsize) {
        num_tiles += 1;
    }

    // Call quantize innermost if the shared exponent axis is the
    // innermost axis and tile size is small
    if (axis == ndim-1 && axis_size % tsize == 0 &&
        tsize <= WARP_SIZE && is_power_of_two(tsize))
    {
        const long total_size = pre_axis_size * axis_size * post_axis_size;
        const long blocks = get_blocks(total_size);
        const int threads = get_threads(total_size);

        if (input.dtype() == torch::ScalarType::Half) {
            AT_ASSERTM(0, " fp16 not supported for MX");
        } else {
            quantize_mx_innermost_cuda_kernel<<<blocks, threads>>>(
                input.data_ptr<float>(),
                scale_bits,
                elem_ebits,
                elem_mbits,
                elem_max_norm,
                total_size,
                tsize,
                flush_fp32_subnorms,
                rounding_mode,
                scale_mode,
                asym,
                output.data_ptr<float>()
            );
        }
    }
    // Otherwise call quantize_mx_by_tile
    else {
        // 1 thread per tile, up to max number of threads
        const long total_tiles = pre_axis_size * num_tiles * post_axis_size;
        const long blocks = get_blocks(total_tiles);
        const int threads = get_threads(total_tiles);

        // Call CUDA kernel
        if (input.dtype() == torch::ScalarType::Half) {
            AT_ASSERTM(0, " fp16 not supported for MX");
        } else {
            quantize_mx_by_tile_cuda_kernel<<<blocks, threads>>>(
                input.data_ptr<float>(),
                scale_bits,
                elem_ebits,
                elem_mbits,
                elem_max_norm,
                total_tiles,
                tsize,
                num_tiles,
                axis_size,
                post_axis_size,
                flush_fp32_subnorms,
                rounding_mode,
                scale_mode,
                asym,
                output.data_ptr<float>()
            );
        }
    }

    gpuErrchk(cudaPeekAtLastError());
    return output;
}


//-----------------------------------------------------------------------
// get_mx_quantize_param_by_tile_cuda
//-----------------------------------------------------------------------
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
) {
    at::Device device = input.device();
    const at::cuda::CUDAGuard device_guard{device};

    const int ndim = input.dim();
    auto input_sizes = input.sizes();

    // 计算维度信息
    const int axis_size = input_sizes[axis];
    int tsize = (tile_size > 0) ? tile_size : axis_size;
    
    long pre_axis_size = 1;
    for (int i = 0; i < axis; i++) {
        pre_axis_size *= input_sizes[i];
    }
    
    long post_axis_size = 1;
    for (int i = axis + 1; i < ndim; i++) {
        post_axis_size *= input_sizes[i];
    }
    
    int num_tiles = axis_size / tsize;
    if (axis_size % tsize) {
        num_tiles += 1;
    }

    const long total_tiles = pre_axis_size * num_tiles * post_axis_size;
    const long total_elems = input.numel();

    // 四个输出张量，shape都与input一致
    auto param_shape = input.sizes();
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto scales1 = torch::zeros(param_shape, options);
    auto scales2 = torch::zeros(param_shape, options);
    auto shifts  = torch::zeros(param_shape, options);
    auto quantized_out = torch::zeros(param_shape, options);
    // // 创建输出张量
    // auto scales_shape = std::vector<long>{total_tiles};
    // auto shifts_shape = std::vector<long>{total_tiles};
    
    // // 对于某些非对称量化模式，可能需要更大的输出张量
    // if (asym == 1) {
    //     scales_shape[0] = total_tiles * 2; // 存储正负缩放因子
    // }
    
    // auto scales1 = torch::zeros(scales_shape, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    // auto scales2 = torch::zeros(scales_shape, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    // auto shifts = torch::zeros(shifts_shape, torch::TensorOptions().dtype(torch::kFloat32).device(device));

    // 计算 CUDA 网格和块配置
    const long blocks = get_blocks(total_tiles);
    const int threads = get_threads(total_tiles);

    // 调用 CUDA 内核
    if (input.dtype() == torch::ScalarType::Half) {
        AT_ASSERTM(0, " fp16 not supported for MX");
    } else {
        get_mx_quantize_param_by_tile_cuda_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(),
            scale_bits,
            elem_ebits,
            elem_mbits,
            elem_max_norm,
            total_tiles,
            tsize,
            num_tiles,
            axis_size,
            post_axis_size,
            flush_fp32_subnorms,
            rounding_mode,
            scale_mode,
            asym,
            scales1.data_ptr<float>(),
            scales2.data_ptr<float>(),
            shifts.data_ptr<float>(),
            quantized_out.data_ptr<float>()   // 新增
        );
    }

    gpuErrchk(cudaPeekAtLastError());
    
    return {scales1, scales2, shifts, quantized_out};
}


//-----------------------------------------------------------------------
// apply_mx_quantize_with_param_cuda
//-----------------------------------------------------------------------
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
) {
    at::Device device = input.device();
    const at::cuda::CUDAGuard device_guard{device};
    auto output = torch::empty_like(input);
    output = output.to(device);

    const int ndim = input.dim();
    auto input_sizes = input.sizes();

    // 计算维度信息
    const int axis_size = input_sizes[axis];
    int tsize = (tile_size > 0) ? tile_size : axis_size;
    
    long pre_axis_size = 1;
    for (int i = 0; i < axis; i++) {
        pre_axis_size *= input_sizes[i];
    }
    
    long post_axis_size = 1;
    for (int i = axis + 1; i < ndim; i++) {
        post_axis_size *= input_sizes[i];
    }
    
    int num_tiles = axis_size / tsize;
    if (axis_size % tsize) {
        num_tiles += 1;
    }

    const long total_tiles = pre_axis_size * num_tiles * post_axis_size;

    // 计算 CUDA 网格和块配置
    const long blocks = get_blocks(total_tiles);
    const int threads = get_threads(total_tiles);

    // 调用 CUDA 内核
    if (input.dtype() == torch::ScalarType::Half) {
        AT_ASSERTM(0, " fp16 not supported for MX");
    } else {
        apply_mx_quantize_with_param_cuda_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(),
            scales1.data_ptr<float>(),
            scales2.data_ptr<float>(),
            shifts.data_ptr<float>(),
            elem_ebits,
            elem_mbits,
            elem_max_norm,
            total_tiles,
            tsize,
            num_tiles,
            axis_size,
            post_axis_size,
            flush_fp32_subnorms,
            rounding_mode,
            scale_mode,
            asym,
            output.data_ptr<float>()
        );
    }

    gpuErrchk(cudaPeekAtLastError());
    return output;
}