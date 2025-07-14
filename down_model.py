import transformers
model_list = [
    "Qwen/Qwen3-8B",
    "Qwen/Qwen-7B"
    # "Qwen/Qwen2-7B",
    # "Qwen/Qwen2-1.5B",
    # "Qwen/Qwen2.5-7B",
    # "Qwen/Qwen2.5-1.5B",
    # # "Qwen/Qwen2.5-72B",
    # "Qwen/Qwen2.5-14B",
    # "Qwen/Qwen2.5-32B",
]
for model_name in model_list:
    print(f"Loading model: {model_name}")
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, device_map="cpu", trust_remote_code=True, torch_dtype="auto")
        print(f"Successfully loaded {model_name}")
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print(f"Tokenizer loaded for {model_name}")
        del model, tokenizer
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
