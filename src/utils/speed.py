import torch, os
def speed_setup(channels_last: bool, cudnn_benchmark: bool):
    if channels_last: torch.backends.cuda.matmul.allow_tf32 = True  # pair w/ next line
    torch.set_float32_matmul_precision("high")  # Ampere+ speedup
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)  # algo autotune for fixed shapes
    # Optional: model.to(memory_format=torch.channels_last) per-module
