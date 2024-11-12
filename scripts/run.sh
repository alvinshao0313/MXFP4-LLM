model=none
seed=0
tasks=piqa
num_fewshot=none
eval_ppl=true

w_elem_format_linear=none
a_elem_format_linear=none
scale_bits_linear=8
block_size_linear=32

A_elem_format_matmul=none
B_elem_format_matmul=none
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
a_scale_mode=0
w_scale_mode=0 
A_scale_mode=0 
B_scale_mode=0 
per_tensor=false

quarot=true
rotate_mode=hadamard
rotate_kv=true
kv_quant_only=false
kv_tokenwise=false

for model in $2
do
for scale_bits in 8
do
scale_bits_linear=$scale_bits
scale_bits_matmul=$scale_bits
for per_tensor in false
do
for block_size in 32
do
block_size_linear=$block_size
block_size_matmul=$block_size
for format in fp4_e2m1_asym
do
w_elem_format_linear=$format
A_elem_format_matmul=$format
a_elem_format_linear=$format
B_elem_format_matmul=$format
for scale_mode in 2
do
w_scale_mode=$scale_mode
A_scale_mode=$scale_mode
a_scale_mode=$scale_mode
B_scale_mode=$scale_mode

CUDA_VISIBLE_DEVICES=$1 python main.py \
    --model=$model \
    --seed=$seed \
    --tasks=$tasks \
    --num_fewshot=$num_fewshot \
    --eval_ppl=$eval_ppl \
    --w_elem_format_linear=$w_elem_format_linear \
    --a_elem_format_linear=$a_elem_format_linear \
    --scale_bits_linear=$scale_bits_linear \
    --block_size_linear=$block_size_linear \
    --A_elem_format_matmul=$A_elem_format_matmul \
    --B_elem_format_matmul=$B_elem_format_matmul \
    --scale_bits_matmul=$scale_bits_matmul \
    --block_size_matmul=$block_size_matmul \
    --w_elem_format_ln=$w_elem_format_ln \
    --a_elem_format_ln=$a_elem_format_ln \
    --scale_bits_ln=$scale_bits_ln \
    --block_size_ln=$block_size_ln \
    --w_elem_format_head=$w_elem_format_head \
    --a_elem_format_head=$a_elem_format_head \
    --scale_bits_head=$scale_bits_head \
    --block_size_head=$block_size_head \
    --auto_dtype=$auto_dtype \
    --custom_cuda=$custom_cuda \
    --a_scale_mode=$a_scale_mode \
    --w_scale_mode=$w_scale_mode \
    --A_scale_mode=$A_scale_mode \
    --B_scale_mode=$B_scale_mode \
    --per_tensor=$per_tensor \
    --quarot=$quarot \
    --rotate_mode=$rotate_mode \
    --rotate_kv=$rotate_kv \
    --kv_quant_only=$kv_quant_only \
    --kv_tokenwise=$kv_tokenwise
done
done
done
done
done
done
