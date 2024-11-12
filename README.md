# Reduced-precision MX-format Framework for LLM inference
## Getting Started
Our experiments are tested on A100 + CUDA Toolkit 11.8 + PyTorch 2.1.2 
```sh
git clone https://github.com/aiha-lab/MX-QLLM.git 
cd MX-QLLM && bash setup.sh
```

## Shared Scale and Element Format
**Set Shared Scale**
- PoT (MX Default Setting): ```scale_mode=0```
- PoT-R: ```scale_mode=3```
- FP16: ```scale_mode=2```
- FP8 (E5M2): ```scale_mode=152```
- FP8 (E4M3): ```scale_mode=143```

**Set Element Format**
- FP8 (E4M3): ```fp8_e4m3```
- FP6 (E3M2): ```fp6_e3m2```
- FP4 (E2M1): ```fp4_e2m1```
- AsymFP4 (E2M1): ```fp4_e2m1_asym```
- INT4: ```int4```
- AsymINT4: ```int4_asym```

## Example Usage
All arguments are in ```scripts/run.sh```
```sh
bash scripts/run.sh [DEVICE_NUM] [MODEL_PATH]
# e.g. bash scripts/run.sh 0 LLMDIR/llama2-7b
```

**MXFP4-PoT**
```sh
...
for format in fp4_e2m1
do
...
for scale_mode in 0
do
...
```

**AMXFP4-FP8**
```sh
...
for format in fp4_e2m1_asym
do
...
for scale_mode in 152
do
...
```

**AMXFP4-FP8 with Randomized Hadamard Rotation**
```sh
...
quarot=true
rotate_mode=hadamard
rotate_kv=true
kv_quant_only=false
kv_tokenwise=false
...
for format in fp4_e2m1_asym
do
...
for scale_mode in 152
do
...
```

## References
**MX Pytorch Emulation Library (https://github.com/microsoft/microxcaling)**
```bib
@misc{rouhani2023microscalingdataformatsdeep,
      title={Microscaling Data Formats for Deep Learning}, 
      author={Bita Darvish Rouhani and Ritchie Zhao and Ankit More and Mathew Hall and Alireza Khodamoradi and Summer Deng and Dhruv Choudhary and Marius Cornea and Eric Dellinger and Kristof Denolf and Stosic Dusan and Venmugil Elango and Maximilian Golub and Alexander Heinecke and Phil James-Roxby and Dharmesh Jani and Gaurav Kolhe and Martin Langhammer and Ada Li and Levi Melnick and Maral Mesmakhosroshahi and Andres Rodriguez and Michael Schulte and Rasoul Shafipour and Lei Shao and Michael Siu and Pradeep Dubey and Paulius Micikevicius and Maxim Naumov and Colin Verrilli and Ralph Wittig and Doug Burger and Eric Chung},
      year={2023},
      eprint={2310.10537},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2310.10537}, 
}
```
**QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs (https://github.com/spcl/QuaRot)**
```bib
@article{ashkboos2024quarot,
  title={QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs},
  author={Ashkboos, Saleh and Mohtashami, Amirkeivan and Croci, Maximilian L and Li, Bo and Jaggi, Martin and Alistarh, Dan and Hoefler, Torsten and Hensman, James},
  journal={arXiv preprint arXiv:2404.00456},
  year={2024}
}
```