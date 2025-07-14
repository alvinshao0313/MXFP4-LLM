import numpy as np
import torch
import argparse
import logging
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    elif v.lower() in ('None'):
        return None
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2list(v):
    if v is None or v.lower() in ('none'):
        return []
    vv = v.split(',')
    ret = []
    for vvv in vv:
        ret.append(vvv)
    return ret


def str2intlist(v):
    vv = v.split(',')
    ret = []
    for vvv in vv:
        ret.append(int(vvv))
    return ret


def str2int(v):
    if v.lower() in ('none'):
        return None
    else:
        return int(v)


def str2path(v):
    if v is None or v.lower() in ('none'):
        return None
    else:
        return str(v)


def cleanup_memory(verbos=True) -> None:
    """Run GC and clear GPU memory."""
    import gc
    import inspect
    caller_name = ''
    try:
        caller_name = f' (from {inspect.stack()[1].function})'
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))

    memory_before = total_reserved_mem()

    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        if verbos:
            logging.info(
                f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
                f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
            )


def distribute_model(model) -> None:
    """Distribute the model across available GPUs. NB: only implemented for Llama-2."""
    from scale_utils import model_utils
    if model_utils.get_model_type(model) == model_utils.LLAMA_MODEL:
        no_split_module_classes = ['LlamaDecoderLayer']
    elif model_utils.get_model_type(model) == model_utils.MISTRAL_MODEL:
        no_split_module_classes = ['MistralDecoderLayer']
    elif model_utils.get_model_type(model) == model_utils.QWEN2_MODEL:
        no_split_module_classes = ['Qwen2DecoderLayer']
    elif model_utils.get_model_type(model) == model_utils.QWEN3_MODEL:
        no_split_module_classes = ['Qwen3DecoderLayer']
    else:
        raise ValueError(f"Unsupported model type: {model_utils.get_model_type(model)}")
    max_memory = get_balanced_memory(
        model,
        no_split_module_classes=no_split_module_classes,
    )

    device_map = infer_auto_device_map(
        model, max_memory=max_memory, no_split_module_classes=no_split_module_classes
    )

    dispatch_model(
        model,
        device_map=device_map,
        offload_buffers=True,
        offload_dir="offload",
        state_dict=model.state_dict(),
    )

    cleanup_memory()
