#!/bin/bash
w_elem_format_linear=none
a_elem_format_linear=none
scale_bits_linear=8
block_size_linear=32
w_elem_format_matmul=none
a_elem_format_matmul=none
scale_bits_matmul=8
block_size_matmul=32
w_elem_format_ln=none
a_elem_format_ln=none
scale_bits_ln=8
block_size_ln=32
w_elem_format_head=none
a_elem_format_head=none
scale_bits_head=8
block_size_head=32
auto_dtype=true
custom_cuda=true
except_layers=lm_head
scale_mode=2 # 0: PoT Floor, 1: PoT Ceil, 2: FP scale, 3: PoT Round 143, 152
per_tensor=false
exhaustive_mixed_format=none # int4_asym,fp4_e2m1_asym
kv_quant_only=true
kv_tokenwise=true
lloyd_max=false
lloyd_max_a=4
lloyd_max_w=4
lloyd_max_kv=4
nclusters=8
lloyd_max_path=none
track_stats=false

for scale_bits in 5
do
for per_tensor in false
do
for block_size in 32
do
for format in fp4_e2m1_asym
do
format_w=$format
format_a=$format
for scale_mode in 1111
do
w_elem_format_linear=$format_w
a_elem_format_linear=$format_a
scale_bits_linear=$scale_bits
block_size_linear=$block_size
w_elem_format_matmul=$format_w
a_elem_format_matmul=$format_a
scale_bits_matmul=$scale_bits
block_size_matmul=$block_size
CUDA_VISIBLE_DEVICES=$1 python run_summarization.py \
    --model_name_or_path facebook/bart-large-cnn \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir ./output_logs/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=32 \
    --overwrite_output_dir \
    --predict_with_generate \
    --w_elem_format_linear $w_elem_format_linear \
    --a_elem_format_linear $a_elem_format_linear \
    --scale_bits_linear $scale_bits_linear \
    --block_size_linear $block_size_linear \
    --w_elem_format_matmul $w_elem_format_matmul \
    --a_elem_format_matmul $a_elem_format_matmul \
    --scale_bits_matmul $scale_bits_matmul \
    --block_size_matmul $block_size_matmul \
    --w_elem_format_ln $w_elem_format_ln \
    --a_elem_format_ln $a_elem_format_ln \
    --scale_bits_ln $scale_bits_ln \
    --block_size_ln $block_size_ln \
    --w_elem_format_head $w_elem_format_head \
    --a_elem_format_head $a_elem_format_head \
    --scale_bits_head $scale_bits_head \
    --block_size_head $block_size_head \
    --auto_dtype $auto_dtype \
    --custom_cuda $custom_cuda \
    --except_layers $except_layers \
    --scale_mode $scale_mode \
    --per_tensor $per_tensor \
    --exhaustive_mixed_format $exhaustive_mixed_format \
    --kv_quant_only $kv_quant_only \
    --kv_tokenwise $kv_tokenwise \
    --lloyd_max $lloyd_max \
    --track_stats $track_stats \
    --lloyd_max_a $lloyd_max_a \
    --lloyd_max_w $lloyd_max_w \
    --lloyd_max_kv $lloyd_max_kv \
    --nclusters $nclusters \
    --lloyd_max_path $lloyd_max_path
done
done
done
done
done
