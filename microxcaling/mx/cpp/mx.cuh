/*
 * Microsoft Confidential
 */

#ifndef PYT_MX_MX_CUH
#define PYT_MX_MX_CUH

#include "common.cuh"
#include "shared_exp.cuh"
#include "quantize.cuh"
#include <math.h>

//-----------------------------------------------------------------------
// quantize_mx_cuda_kernel
//-----------------------------------------------------------------------
template<typename T>
__global__ void quantize_mx_cuda_kernel(
    const T* __restrict__ input,
    const int scale_bits,
    int elem_ebits,
    int elem_mbits,
    float elem_max_norm,
    const float* __restrict__ max_values,
    const float* __restrict__ pos_values,
    const float* __restrict__ neg_values,
    const float* __restrict__ std_values,
    const long total_size,
    const int axis_size,
    const int post_axis_size,
    const bool flush_fp32_subnorms,
    const RoundingMode rounding_mode,
    const int scale_mode,
    const int asym,
    T* __restrict__ output
) {
    const long offset = blockDim.x * blockIdx.x + threadIdx.x;
    if (offset >= total_size) return;

    // Compute index of the max value for this element
    const long post_axis_i = offset % post_axis_size;
    const long pre_axis_i = offset / (post_axis_size * axis_size);

    // Get shared exponent
    const long m_i = pre_axis_i * post_axis_size + post_axis_i;
    if (asym>=0){
        if (asym==0){
            float scale = (pos_values[m_i] - neg_values[m_i]) / ((1 << elem_mbits) - 1);
            scale = (scale==0) ? 1 : scale; // Avoid zero division
            float shift = -1 * (round(neg_values[m_i]/scale)) - (1<<(elem_mbits-1));
            T scaled_in = input[offset] / scale + shift;
            T scaled_out = round(scaled_in);
            output[offset] = (scaled_out-shift) * scale;
        } else if (asym==1){
            float scale_pos = pos_values[m_i] / elem_max_norm;
            float scale_neg = fabsf(neg_values[m_i]) / elem_max_norm;
            scale_pos = (scale_pos==0) ? 1 : scale_pos; // Avoid zero division
            scale_neg = (scale_neg==0) ? 1 : scale_neg; // Avoid zero division
            float scale = (input[offset]>0) ? scale_pos : scale_neg;
            if (scale_mode==143) { // get fp max-scale from given tensor
                scale = quantize_elemwise(
                        scale, 5, 4, 480,
                        rounding_mode, true, true);
                scale = (scale==0) ? 1 : scale;
            } else if (scale_mode==152) { // get fp max-scale from given tensor
                scale = quantize_elemwise(
                        scale, 4, 5, 57344.0,
                        rounding_mode, true, true);
            }
            T scaled_in = input[offset] / scale;
            T scaled_out = quantize_elemwise(
                    scaled_in, elem_mbits, elem_ebits, elem_max_norm,
                    rounding_mode, true, true);
            output[offset] = scaled_out * scale;
        } else if (asym==3){
            float scale = (pos_values[m_i] - neg_values[m_i]) / ((1 << elem_mbits) - 1);
            scale = (scale==0) ? 1 : scale; // Avoid zero division
            scale = (scale_mode==3) ? powf(2.0f, roundf(log2f(scale))) : powf(2.0f, floorf(log2f(scale)));
            float shift = -1 * (round(neg_values[m_i]/scale)) - (1<<(elem_mbits-1));
            T scaled_in = input[offset] / scale + shift;
            T scaled_out = round(scaled_in);
            T max_clip = (1 << (elem_mbits-1)) - 1; // -8~7
            scaled_out = fminf(max_clip, scaled_out);
            output[offset] = (scaled_out-shift) * scale;
        } else if (asym==2){ // PoT
            int shared_exp_pos = (int) get_biased_exponent(pos_values[m_i]);
            int shared_exp_neg = (int) get_biased_exponent(neg_values[m_i]);
            int shared_exp = (input[offset]>0) ? shared_exp_pos : shared_exp_neg;
            if (scale_mode==3) { // Ceil if necessary
                int threshold = 0x7FFFFF;
                threshold >>= (24 - elem_mbits);
                threshold <<= (24 - elem_mbits);
                int mantissa = (input[offset]>0) ? (*(int*)&pos_values[m_i] & 0x7FFFFF) : (*(int*)&neg_values[m_i] & 0x7FFFFF);
                if (mantissa >= threshold) {
                    shared_exp += 1;
                }
            }
            float scale = mx_get_shared_scale(
                  shared_exp, scale_bits, elem_max_norm);
            T scaled_in = input[offset] / scale;
            T scaled_out = quantize_elemwise(
                    scaled_in, elem_mbits, elem_ebits, elem_max_norm,
                    rounding_mode, true, true);
            output[offset] = scaled_out * scale;
        } else if (asym==4){ // PoT x Synced mantissa
            int shared_exp_pos = (int) get_biased_exponent(pos_values[m_i]);
            int shared_exp_neg = (int) get_biased_exponent(neg_values[m_i]);
            int shared_exp = (input[offset]>0) ? shared_exp_pos : shared_exp_neg;
            float scale = mx_get_shared_scale(
                  shared_exp, scale_bits, elem_max_norm);
            int raw_mantissa = 0;
            float mantissa = 1.0;
            if (scale_mode==11) {
                raw_mantissa = (*(int*)&max_values[m_i] & 0x600000) >> 21;
                mantissa = 1.0 + (float)raw_mantissa / 4.0;
            } else if (scale_mode==111) {
                raw_mantissa = (*(int*)&max_values[m_i] & 0x700000) >> 20;
                mantissa = 1.0 + (float)raw_mantissa / 8.0;
            } else if (scale_mode==1111) {
                raw_mantissa = (*(int*)&max_values[m_i] & 0x780000) >> 19;
                mantissa = 1.0 + (float)raw_mantissa / 16.0;
            } else if (scale_mode==11111) {
                raw_mantissa = (*(int*)&max_values[m_i] & 0x7C0000) >> 18;
                mantissa = 1.0 + (float)raw_mantissa / 32.0;
            } else if (scale_mode==111111) {
                raw_mantissa = (*(int*)&max_values[m_i] & 0x7E0000) >> 17;
                mantissa = 1.0 + (float)raw_mantissa / 64.0;
            } else if (scale_mode==1111111) {
                raw_mantissa = (*(int*)&max_values[m_i] & 0x7F0000) >> 16;
                mantissa = 1.0 + (float)raw_mantissa / 128.0;
            }
            scale = scale*mantissa;
            T scaled_in = input[offset] / scale;
            T scaled_out = quantize_elemwise(
                    scaled_in, elem_mbits, elem_ebits, elem_max_norm,
                    rounding_mode, true, true);
            output[offset] = scaled_out * scale;
        }
    } else{
        int shared_exp = (int) get_biased_exponent(max_values[m_i]);
        bool flush_tile = (shared_exp == 0 && flush_fp32_subnorms);
    
        // Compute the shared scale
        if (scale_mode==1) {
            shared_exp += 1;
        }
        if (scale_mode==3) { // Ceil if necessary
            int threshold = 0x7FFFFF;
            threshold >>= (24 - elem_mbits);
            threshold <<= (24 - elem_mbits);
            int mantissa = (*(int*)&max_values[m_i] & 0x7FFFFF);
            if (mantissa >= threshold) {
                shared_exp += 1;
            }
        }
        float scale = 1;
        if (scale_mode==2) { // get fp max-scale from given tensor
            scale = max_values[m_i] / elem_max_norm;
            scale = (scale==0) ? 1 : scale;
        } else if (scale_mode==143) { // get fp max-scale from given tensor
            scale = max_values[m_i] / elem_max_norm;
            scale = quantize_elemwise(
                    scale, 5, 4, 480,
                    rounding_mode, true, true);
            scale = (scale==0) ? 1 : scale;
        } else if (scale_mode==152) { // get fp max-scale from given tensor
            scale = max_values[m_i] / elem_max_norm;
            scale = quantize_elemwise(
                    scale, 4, 5, 57344.0,
                    rounding_mode, true, true);
            scale = (scale==0) ? 1 : scale;
        } else{
            scale = mx_get_shared_scale(
                  shared_exp, scale_bits, elem_max_norm);
        }
    
        T scaled_in = (flush_tile) ? 0 : input[offset] / scale;
    
        T scaled_out = quantize_elemwise(
                scaled_in, elem_mbits, elem_ebits, elem_max_norm,
                rounding_mode, true, true);
    
        output[offset] = scaled_out * scale;
    }
}

//-----------------------------------------------------------------------
// quantize_innermost, fast MX quantization for axis=[-1]
// input requirements:
//  - the axis is dim-1 (the innermost dim),
//  - tile_size divides axis_size evenly
//  - tile_size is a power of 2
//  - tile_size <= WARP_SIZE
//-----------------------------------------------------------------------
template<typename T>
__global__ void quantize_mx_innermost_cuda_kernel (
    const T* __restrict__ in,
    const int scale_bits,
    int elem_ebits,
    int elem_mbits,
    float elem_max_norm,
    const long total_size,
    const int tile_size,
    const bool flush_fp32_subnorms,
    const RoundingMode rounding_mode,
    const int scale_mode,
    const int asym,
    T* __restrict__ out
) {
    const long offset = blockDim.x * blockIdx.x + threadIdx.x;
    if (offset >= total_size) return;
    const T elem = in[offset];

    if (asym>=0) { 
        if (asym==0){ // Asymmetric quantization for INT format
            float max_ = elem;
            float min_ = elem;
            for (int mask = tile_size/2; mask > 0; mask /= 2) {
                float _tmp_max = __shfl_xor_sync(0xFFFFFFFF, max_, mask);
                float _tmp_min = __shfl_xor_sync(0xFFFFFFFF, min_, mask);
                max_ = (_tmp_max > max_) ? _tmp_max : max_;
                min_ = (_tmp_min < min_) ? _tmp_min : min_;
            }
            float scale = (max_ - min_) / ((1 << elem_mbits) - 1);
            scale = (scale==0) ? 1 : scale;
            float shift = -1 * (round(min_/scale)) - (1<<(elem_mbits-1));
            T scaled_in = elem / scale + shift;
            T scaled_out = round(scaled_in);
            out[offset] = (scaled_out-shift) * scale;
        } else if (asym==3){ // AsymInt PoT
            float max_ = elem;
            float min_ = elem;
            for (int mask = tile_size/2; mask > 0; mask /= 2) {
                float _tmp_max = __shfl_xor_sync(0xFFFFFFFF, max_, mask);
                float _tmp_min = __shfl_xor_sync(0xFFFFFFFF, min_, mask);
                max_ = (_tmp_max > max_) ? _tmp_max : max_;
                min_ = (_tmp_min < min_) ? _tmp_min : min_;
            }
            float scale = (max_ - min_) / ((1 << elem_mbits) - 1);
            scale = (scale==0) ? 1 : scale;
            scale = (scale_mode==3) ? powf(2.0f, roundf(log2f(scale))) : powf(2.0f, floorf(log2f(scale)));
            float shift = -1 * (round(min_/scale)) - (1<<(elem_mbits-1));
            T scaled_in = elem / scale + shift;
            T scaled_out = round(scaled_in);
            T max_clip = (1 << (elem_mbits-1)) - 1; // -8~7
            scaled_out = fmaxf(0, fminf(max_clip, scaled_out));
            out[offset] = (scaled_out-shift) * scale;
        } else if (asym==1) { // Asymmetric quantization for Floating-Point format
            float max_ = elem;
            float min_ = elem;
            for (int mask = tile_size/2; mask > 0; mask /= 2) {
                float _tmp_max = __shfl_xor_sync(0xFFFFFFFF, max_, mask);
                float _tmp_min = __shfl_xor_sync(0xFFFFFFFF, min_, mask);
                max_ = (_tmp_max > max_) ? _tmp_max : max_;
                min_ = (_tmp_min < min_) ? _tmp_min : min_;
            }
            float scale_pos = max_ / elem_max_norm;
            float scale_neg = fabsf(min_) / elem_max_norm;
            scale_pos = (scale_pos==0) ? 1 : scale_pos;
            scale_neg = (scale_neg==0) ? 1 : scale_neg;
            float scale = (elem>0) ? scale_pos : scale_neg;
            if (scale_mode==143) {
                scale = quantize_elemwise(
                        scale, 5, 4, 480,
                        rounding_mode, true, true);
                scale = (scale==0) ? 1 : scale;
            } else if (scale_mode==152) {
                scale = quantize_elemwise(
                        scale, 4, 5, 57344.0,
                        rounding_mode, true, true);
            }
            T scaled_in = elem / scale;
            T scaled_out = quantize_elemwise(
                    scaled_in, elem_mbits, elem_ebits, elem_max_norm,
                    rounding_mode, true, true);
            out[offset] = scaled_out * scale;
        } else if (asym==2) { // AsymFloat PoT
            float max_ = elem;
            float min_ = elem;
            for (int mask = tile_size/2; mask > 0; mask /= 2) {
                float _tmp_max = __shfl_xor_sync(0xFFFFFFFF, max_, mask);
                float _tmp_min = __shfl_xor_sync(0xFFFFFFFF, min_, mask);
                max_ = (_tmp_max > max_) ? _tmp_max : max_;
                min_ = (_tmp_min < min_) ? _tmp_min : min_;
            }
            int shared_exp_pos = get_biased_exponent(max_);
            int shared_exp_neg = get_biased_exponent(min_);
            int shared_exp = (elem>0) ? shared_exp_pos : shared_exp_neg;
            if (scale_mode==3) { // Ceil if necessary
                int threshold = 0x7FFFFF;
                threshold >>= (24 - elem_mbits);
                threshold <<= (24 - elem_mbits);
                int mantissa = (elem>0) ? (*(int*)&max_ & 0x7FFFFF) : (*(int*)&min_ & 0x7FFFFF);
                if (mantissa >= threshold) {
                    shared_exp += 1;
                }
            }
            float scale = mx_get_shared_scale(
                  shared_exp, scale_bits, elem_max_norm);
            T scaled_in = elem / scale;
            T scaled_out = quantize_elemwise(
                    scaled_in, elem_mbits, elem_ebits, elem_max_norm,
                    rounding_mode, true, true);
            out[offset] = scaled_out * scale;
        } else if (asym==4) { // AsymFloat PoT
            float max_ = elem;
            float min_ = elem;
            float max_val = fabsf(elem); // absolute max
            for (int mask = tile_size/2; mask > 0; mask /= 2) {
                float _tmp_max = __shfl_xor_sync(0xFFFFFFFF, max_, mask);
                float _tmp_min = __shfl_xor_sync(0xFFFFFFFF, min_, mask);
                float _tmp_elem = __shfl_xor_sync(0xFFFFFFFF, max_val, mask);
                max_ = (_tmp_max > max_) ? _tmp_max : max_;
                min_ = (_tmp_min < min_) ? _tmp_min : min_;
                max_val = (_tmp_elem > max_val) ? _tmp_elem : max_val;
            }
            int shared_exp_pos = get_biased_exponent(max_);
            int shared_exp_neg = get_biased_exponent(min_);
            int shared_exp = (elem>0) ? shared_exp_pos : shared_exp_neg;
            float scale = mx_get_shared_scale(
                  shared_exp, scale_bits, elem_max_norm);
            int raw_mantissa = 0;
            float mantissa = 1.0;
            if (scale_mode==11) {
                raw_mantissa = (*(int*)&max_val & 0x600000) >> 21;
                mantissa = 1.0 + (float)raw_mantissa / 4.0;
            } else if (scale_mode==111) {
                raw_mantissa = (*(int*)&max_val & 0x700000) >> 20;
                mantissa = 1.0 + (float)raw_mantissa / 8.0;
            } else if (scale_mode==1111) {
                raw_mantissa = (*(int*)&max_val & 0x780000) >> 19;
                mantissa = 1.0 + (float)raw_mantissa / 16.0;
            } else if (scale_mode==11111) {
                raw_mantissa = (*(int*)&max_val & 0x7C0000) >> 18;
                mantissa = 1.0 + (float)raw_mantissa / 32.0;
            } else if (scale_mode==111111) {
                raw_mantissa = (*(int*)&max_val & 0x7E0000) >> 17;
                mantissa = 1.0 + (float)raw_mantissa / 64.0;
            } else if (scale_mode==1111111) {
                raw_mantissa = (*(int*)&max_val & 0x7F0000) >> 16;
                mantissa = 1.0 + (float)raw_mantissa / 128.0;
            }
            scale = scale*mantissa;
            T scaled_in = elem / scale;
            T scaled_out = quantize_elemwise(
                    scaled_in, elem_mbits, elem_ebits, elem_max_norm,
                    rounding_mode, true, true);
            out[offset] = scaled_out * scale;
        }
    } else{
        // allreduce to get the max value in each tile
        int shared_exp = get_biased_exponent(elem);
        float max_val = fabsf(elem); // absolute max
        for (int mask = tile_size/2; mask > 0; mask /= 2) {
            int _tmp = __shfl_xor_sync(0xFFFFFFFF, shared_exp, mask);
            shared_exp = (_tmp > shared_exp) ? _tmp : shared_exp;
            // Compare value from other thread in the warp
            float _tmp_elem = __shfl_xor_sync(0xFFFFFFFF, max_val, mask);
            max_val = (_tmp_elem > max_val) ? _tmp_elem : max_val;
        }
    
        bool flush_tile = (shared_exp == 0 && flush_fp32_subnorms);
    
        // Compute the shared scale
        if (scale_mode==1) { // Ceil instead of floor
            int threshold = 0;
            int mantissa = (*(int*)&max_val & 0x7FFFFF);
            if (mantissa > threshold) {
                shared_exp += 1;
            }
        }
        if (scale_mode==3) { // Ceil if necessary
            int threshold = 0x7FFFFF;
            threshold >>= (24 - elem_mbits);
            threshold <<= (24 - elem_mbits);
            int mantissa = (*(int*)&max_val & 0x7FFFFF);
            if (mantissa >= threshold) {
                shared_exp += 1;
            }
        }
        float scale = 1;
        if (scale_mode==2) {
            scale = max_val / elem_max_norm;
            scale = (scale==0) ? 1 : scale;
        } else if (scale_mode==143) {
            scale = max_val / elem_max_norm;
            scale = quantize_elemwise(
                    scale, 5, 4, 480,
                    rounding_mode, true, true);
            scale = (scale==0) ? 1 : scale;
        } else if (scale_mode==152) {
            scale = max_val / elem_max_norm;
            scale = quantize_elemwise(
                    scale, 4, 5, 57344.0,
                    rounding_mode, true, true);
        } else{
            scale = mx_get_shared_scale(
                  shared_exp, scale_bits, elem_max_norm);
        }
    
        T scaled_in = (flush_tile) ? 0 : elem / scale;
    
        T scaled_out = quantize_elemwise(
                scaled_in, elem_mbits, elem_ebits, elem_max_norm,
                rounding_mode, true, true);
    
        out[offset] = scaled_out * scale;
    }
}

//-----------------------------------------------------------------------
// quantize_mx_by_tile kernel
// Each thread loops across the tile to get the max exponent, then
// loops across it again to perform quantization.
//-----------------------------------------------------------------------
template<typename T>
__global__ void quantize_mx_by_tile_cuda_kernel (
    const T* __restrict__ in,
    const int scale_bits,
    int elem_ebits,
    int elem_mbits,
    float elem_max_norm,
    const int total_tiles,
    const int tile_size,
    const int num_tiles,
    const int axis_size,
    const int post_axis_size,
    const bool flush_fp32_subnorms,
    const RoundingMode rounding_mode,
    const int scale_mode,
    const int asym,
    T* __restrict__ out
) {
    const long offset = blockDim.x * blockIdx.x + threadIdx.x;
    if (offset >= total_tiles) return;

    // Calculate indices on different dimensions
    const long post_axis_i = offset % post_axis_size;
    const long num_tiles_i = (offset / post_axis_size) % num_tiles;
    const long pre_axis_i = offset / (num_tiles * post_axis_size);

    // Handle non-full bounding box/tile
    int adjusted_tile_size;
    if ((num_tiles_i + 1) * tile_size > axis_size) {
        adjusted_tile_size = axis_size % tile_size;
    } else {
        adjusted_tile_size = tile_size;
    }

    if (asym>=0) { 
        if (asym==0) { // Asymmetric quantization for INT format
            float max_ = 0;
            float min_ = 0;
            for (int i = 0; i < adjusted_tile_size; i++) {
                long in_i = pre_axis_i * axis_size * post_axis_size +
                    (num_tiles_i * tile_size + i) * post_axis_size +
                    post_axis_i;
                max_ = (in[in_i] > max_) ? in[in_i] : max_;
                min_ = (in[in_i] < min_) ? in[in_i] : min_;
            }
            float scale = (max_ - min_) / ((1 << elem_mbits) - 1);
            scale = (scale==0) ? 1 : scale;
            float shift = -1 * (round(min_/scale)) - (1<<(elem_mbits-1));
            // Loop over bounding box to quantize
            for (int i = 0; i < adjusted_tile_size; i++) {
                long in_i = pre_axis_i * axis_size * post_axis_size +
                    (num_tiles_i * tile_size + i) * post_axis_size +
                    post_axis_i;
        
                T scaled_in = in[in_i] / scale + shift;
                T scaled_out = round(scaled_in);
                out[in_i] = (scaled_out-shift) * scale;
            }
        } else if (asym==3) { // Asymmetric quantization for INT format
            float max_ = 0;
            float min_ = 0;
            for (int i = 0; i < adjusted_tile_size; i++) {
                long in_i = pre_axis_i * axis_size * post_axis_size +
                    (num_tiles_i * tile_size + i) * post_axis_size +
                    post_axis_i;
                max_ = (in[in_i] > max_) ? in[in_i] : max_;
                min_ = (in[in_i] < min_) ? in[in_i] : min_;
            }
            float scale = (max_ - min_) / ((1 << elem_mbits) - 1);
            scale = (scale==0) ? 1 : scale;
            scale = (scale_mode==3) ? powf(2.0f, roundf(log2f(scale))) : powf(2.0f, floorf(log2f(scale)));
            float shift = -1 * (round(min_/scale)) - (1<<(elem_mbits-1));
            T max_clip = (1 << (elem_mbits-1)) - 1; // -8~7
            // Loop over bounding box to quantize
            for (int i = 0; i < adjusted_tile_size; i++) {
                long in_i = pre_axis_i * axis_size * post_axis_size +
                    (num_tiles_i * tile_size + i) * post_axis_size +
                    post_axis_i;
        
                T scaled_in = in[in_i] / scale + shift;
                T scaled_out = round(scaled_in);
                scaled_out = fmaxf(0, fminf(max_clip, scaled_out));
                out[in_i] = (scaled_out-shift) * scale;
            }
        } else if (asym==1) {
            float max_ = 0;
            float min_ = 0;
            for (int i = 0; i < adjusted_tile_size; i++) {
                long in_i = pre_axis_i * axis_size * post_axis_size +
                    (num_tiles_i * tile_size + i) * post_axis_size +
                    post_axis_i;
                max_ = (in[in_i] > max_) ? in[in_i] : max_;
                min_ = (in[in_i] < min_) ? in[in_i] : min_;
            }
            float scale_pos = max_ / elem_max_norm;
            float scale_neg = fabsf(min_) / elem_max_norm;
            scale_pos = (scale_pos==0) ? 1 : scale_pos;
            scale_neg = (scale_neg==0) ? 1 : scale_neg;
            float scale = 1;
            // Loop over bounding box to quantize
            for (int i = 0; i < adjusted_tile_size; i++) {
                long in_i = pre_axis_i * axis_size * post_axis_size +
                    (num_tiles_i * tile_size + i) * post_axis_size +
                    post_axis_i;
        
                scale = (in[in_i]>0) ? scale_pos : scale_neg;
                if (scale_mode==143) {
                    scale = quantize_elemwise(
                            scale, 5, 4, 480,
                            rounding_mode, true, true);
                    scale = (scale==0) ? 1 : scale;
                } else if (scale_mode==152) {
                    scale = quantize_elemwise(
                            scale, 4, 5, 57344.0,
                            rounding_mode, true, true);
                }
                T scaled_in = in[in_i] / scale;
        
                T scaled_out = quantize_elemwise(
                        scaled_in, elem_mbits, elem_ebits, elem_max_norm,
                        rounding_mode, true, true);
        
                out[in_i] = scaled_out * scale;
            }
        } else if (asym==2) {
            float max_ = 0;
            float min_ = 0;
            for (int i = 0; i < adjusted_tile_size; i++) {
                long in_i = pre_axis_i * axis_size * post_axis_size +
                    (num_tiles_i * tile_size + i) * post_axis_size +
                    post_axis_i;
                max_ = (in[in_i] > max_) ? in[in_i] : max_;
                min_ = (in[in_i] < min_) ? in[in_i] : min_;
            }
            int shared_exp_pos = get_biased_exponent(max_);
            int shared_exp_neg = get_biased_exponent(min_);
            // Loop over bounding box to quantize
            for (int i = 0; i < adjusted_tile_size; i++) {
                long in_i = pre_axis_i * axis_size * post_axis_size +
                    (num_tiles_i * tile_size + i) * post_axis_size +
                    post_axis_i;
                int shared_exp = (in[in_i]>0) ? shared_exp_pos : shared_exp_neg;
                if (scale_mode==3) { // Ceil if necessary
                    int threshold = 0x7FFFFF;
                    threshold >>= (24 - elem_mbits);
                    threshold <<= (24 - elem_mbits);
                    int mantissa = (in[in_i]>0) ? (*(int*)&max_ & 0x7FFFFF) : (*(int*)&min_ & 0x7FFFFF);
                    if (mantissa >= threshold) {
                        shared_exp += 1;
                    }
                }
                float scale = mx_get_shared_scale(
                      shared_exp, scale_bits, elem_max_norm);
        
                T scaled_in = in[in_i] / scale;
        
                T scaled_out = quantize_elemwise(
                        scaled_in, elem_mbits, elem_ebits, elem_max_norm,
                        rounding_mode, true, true);
        
                out[in_i] = scaled_out * scale;
            }
        } else if (asym==4) {
            float max_ = 0;
            float min_ = 0;
            float max_val = 0;
            for (int i = 0; i < adjusted_tile_size; i++) {
                long in_i = pre_axis_i * axis_size * post_axis_size +
                    (num_tiles_i * tile_size + i) * post_axis_size +
                    post_axis_i;
                max_ = (in[in_i] > max_) ? in[in_i] : max_;
                min_ = (in[in_i] < min_) ? in[in_i] : min_;
                max_val = (fabsf(in[in_i]) > max_val) ? fabsf(in[in_i]) : max_val;
            }
            int shared_exp_pos = get_biased_exponent(max_);
            int shared_exp_neg = get_biased_exponent(min_);
            int raw_mantissa = 0;
            float mantissa = 1.0;
            if (scale_mode==11) {
                raw_mantissa = (*(int*)&max_val & 0x600000) >> 21;
                mantissa = 1.0 + (float)raw_mantissa / 4.0;
            } else if (scale_mode==111) {
                raw_mantissa = (*(int*)&max_val & 0x700000) >> 20;
                mantissa = 1.0 + (float)raw_mantissa / 8.0;
            } else if (scale_mode==1111) {
                raw_mantissa = (*(int*)&max_val & 0x780000) >> 19;
                mantissa = 1.0 + (float)raw_mantissa / 16.0;
            } else if (scale_mode==11111) {
                raw_mantissa = (*(int*)&max_val & 0x7C0000) >> 18;
                mantissa = 1.0 + (float)raw_mantissa / 32.0;
            } else if (scale_mode==111111) {
                raw_mantissa = (*(int*)&max_val & 0x7E0000) >> 17;
                mantissa = 1.0 + (float)raw_mantissa / 64.0;
            } else if (scale_mode==1111111) {
                raw_mantissa = (*(int*)&max_val & 0x7F0000) >> 16;
                mantissa = 1.0 + (float)raw_mantissa / 128.0;
            }
            // Loop over bounding box to quantize
            for (int i = 0; i < adjusted_tile_size; i++) {
                long in_i = pre_axis_i * axis_size * post_axis_size +
                    (num_tiles_i * tile_size + i) * post_axis_size +
                    post_axis_i;
                int shared_exp = (in[in_i]>0) ? shared_exp_pos : shared_exp_neg;
                float scale = mx_get_shared_scale(
                      shared_exp, scale_bits, elem_max_norm);
                scale = scale*mantissa;
        
                T scaled_in = in[in_i] / scale;
        
                T scaled_out = quantize_elemwise(
                        scaled_in, elem_mbits, elem_ebits, elem_max_norm,
                        rounding_mode, true, true);
        
                out[in_i] = scaled_out * scale;
            }
        }
    } else{
        // Find biased shared_exp
        int shared_exp = 0; // biased exp must be >= 0
        float max_val = 0;
        for (int i = 0; i < adjusted_tile_size; i++) {
            long in_i = pre_axis_i * axis_size * post_axis_size +
                (num_tiles_i * tile_size + i) * post_axis_size +
                post_axis_i;
    
            int exp = get_biased_exponent(in[in_i]);
            shared_exp = (exp > shared_exp) ? exp : shared_exp;
            max_val = (fabsf(in[in_i]) > max_val) ? fabsf(in[in_i]) : max_val;
        }
    
        bool flush_tile = (shared_exp == 0 && flush_fp32_subnorms);
    
        // Compute the shared scale
        if (scale_mode==1) {
            int threshold = 0;
            int mantissa = (*(int*)&max_val & 0x7FFFFF);
            if (mantissa > threshold) {
                shared_exp += 1;
            }
            shared_exp += 1;
        }
        if (scale_mode==3) { // Ceil if necessary
            int threshold = 0x7FFFFF;
            threshold >>= (24 - elem_mbits);
            threshold <<= (24 - elem_mbits);
            int mantissa = (*(int*)&max_val & 0x7FFFFF);
            if (mantissa >= threshold) {
                shared_exp += 1;
            }
        }
        float scale = 1;
        if (scale_mode==2) {
            scale = max_val / elem_max_norm;
            scale = (scale==0) ? 1 : scale;
        } else if (scale_mode==143) {
            scale = max_val / elem_max_norm;
            scale = quantize_elemwise(
                    scale, 5, 4, 480,
                    rounding_mode, true, true);
            scale = (scale==0) ? 1 : scale;
        } else if (scale_mode==152) {
            scale = max_val / elem_max_norm;
            scale = quantize_elemwise(
                    scale, 4, 5, 57344.0,
                    rounding_mode, true, true);
        } else{
            scale = mx_get_shared_scale(
                  shared_exp, scale_bits, elem_max_norm);
        }
    
        // Loop over bounding box to quantize
        for (int i = 0; i < adjusted_tile_size; i++) {
            long in_i = pre_axis_i * axis_size * post_axis_size +
                (num_tiles_i * tile_size + i) * post_axis_size +
                post_axis_i;
    
            T scaled_in = (flush_tile) ? 0 : in[in_i] / scale;
    
            T scaled_out = quantize_elemwise(
                    scaled_in, elem_mbits, elem_ebits, elem_max_norm,
                    rounding_mode, true, true);
    
            out[in_i] = scaled_out * scale;
        }
    }
}

#endif
