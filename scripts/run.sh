# Set Shared Scale:
    # PoT (MX Default Setting): scale_mode=0
    # PoT-R: scale_mode=3
    # FP16: scale_mode=2
    # FP8 (E5M2): scale_mode=152
    # FP8 (E4M3): scale_mode=143

# Set Element Format:
    # FP8 (E4M3): fp8_e4m3
    # FP6 (E3M2): fp6_e3m2
    # FP4 (E2M1): fp4_e2m1
    # AsymFP4 (E2M1): fp4_e2m1_asym
    # INT4: int4
    # AsymINT4: int4_asym

model=none
seed=0
tasks=piqa,arc_easy,arc_challenge,winogrande,social_iqa,openbookqa
num_fewshot=none
eval_ppl=true

w_elem_format_linear=none
a_elem_format_linear=none
scale_bits_linear=8
block_size_linear=16

A_elem_format_matmul=none
B_elem_format_matmul=none
scale_bits_matmul=8
block_size_matmul=16

w_elem_format_ln=none
a_elem_format_ln=none
scale_bits_ln=8
block_size_ln=16

w_elem_format_head=none
a_elem_format_head=none
scale_bits_head=8
block_size_head=16

auto_dtype=true
custom_cuda=true
a_scale_mode=0
w_scale_mode=0 
A_scale_mode=0 
B_scale_mode=0 
per_tensor=false

rotate=true
rotate_mode=hadamard
rotate_kv=true
sorting_transform=/home/shaoyuantian/program/MixedPrecision/act_avg/wikitext2_128samples/Meta-Llama-3-8B/sorting_idx/sorted_idx.pt
group_rotate_kv=true
kv_quant_only=true
kv_tokenwise=true

for model in $2
do
for scale_bits in 8
do
scale_bits_linear=$scale_bits
scale_bits_matmul=$scale_bits
for per_tensor in false
do
for block_size in 16
do
block_size_linear=$block_size
block_size_matmul=$block_size
for format in none
do
w_elem_format_linear=$format
A_elem_format_matmul=$format
a_elem_format_linear=$format
B_elem_format_matmul=$format
for scale_mode in 0
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
    --rotate=$rotate \
    --rotate_mode=$rotate_mode \
    --rotate_kv=$rotate_kv \
    --sorting_transform=$sorting_transform \
    --group_rotate_kv=$group_rotate_kv \
    --kv_quant_only=$kv_quant_only \
    --kv_tokenwise=$kv_tokenwise
done
done
done
done
done
done
