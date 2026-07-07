import os
from collections import OrderedDict
from pathlib import Path

import torch


def increment_path(path, exist_ok=False, sep="", mkdir=False):
    path = Path(path)
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")
        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"
            if not os.path.exists(p):
                path = p
                break
    if mkdir:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
    return path


def get_gpu_mem():
    return f"{torch.cuda.memory_reserved() / 1e9:.3g}G" if torch.cuda.is_available() else "0G"


def resolve_device(device):
    device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeError(
            "LCMamNet relies on the Mamba selective-scan CUDA kernels and requires a "
            f"CUDA device; got device='{device}'. CPU execution is not supported."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available, but LCMamNet requires a CUDA device "
            "(the Mamba selective-scan kernels have no CPU implementation)."
        )
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def _extract_state_dict(ckpt):
    if not isinstance(ckpt, dict):
        return None
    for key in ("state_dict", "model"):
        if key in ckpt:
            return ckpt[key]
    return ckpt


def _checkpoint_source(ckpt):
    if not (isinstance(ckpt, dict) and "args" in ckpt):
        return None, None
    args = ckpt["args"]
    if isinstance(args, dict):
        return args.get("name"), args.get("model")
    return getattr(args, "name", None), getattr(args, "model", None)


def _strip_prefix(key):
    for prefix in ("module.", "net.", "model."):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def smart_load_weights(model, weight_path):
    if weight_path is None or not os.path.exists(weight_path):
        print(f"Weight path not found: {weight_path}")
        return model

    print(f"Loading pretrained weights from {weight_path}")
    try:
        ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"Error loading checkpoint {weight_path}: {e}")
        return model

    state_dict = _extract_state_dict(ckpt)
    if state_dict is None:
        print("Unrecognized checkpoint format.")
        return model

    source_name, source_model = _checkpoint_source(ckpt)
    if source_name or source_model:
        print(f"Checkpoint source: name={source_name}, model={source_model}")

    # 只保留与当前模型形状一致的权重，容忍 DataParallel 等前缀差异。
    model_state = model.state_dict()
    matched = OrderedDict()
    skipped = 0
    for k, v in state_dict.items():
        name = _strip_prefix(k)
        if name in model_state and v.shape == model_state[name].shape:
            matched[name] = v
        else:
            skipped += 1

    if matched:
        model.load_state_dict(matched, strict=False)
    if skipped:
        print(f"Skipped layers (unmatched): {skipped}")
    print(f"Transferred {len(matched)}/{len(model_state)} items")
    return model
