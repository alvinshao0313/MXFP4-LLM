# Reduced-precision MX-format Framework for LLM inference
## Getting started
```sh
git clone https://github.com/superdocker/mx-qllm.git  
MX_DIR=/home/hwanii/workspace/mx-qllm
HF_CACHE_DIR=/home/hwanii/hf_cache
docker run -it --rm --gpus all --ipc=host -v $MX_DIR:/root/mx-qllm -v /raid:/raid -v $HF_CACHE_DIR:/hf_cache 166.104.35.43:5000/hwanii/pytorch2.1-cuda11.8:1.2 bash
cd /root/mx-qllm && bash env.sh
```


