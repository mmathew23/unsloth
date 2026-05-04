import torch


# UNSLOTH_TILEGYM_DIFF_REVIEW: TileGym passes torch.cuda.current_stream()
# directly. Unsloth converts it to the raw stream handle accepted reliably by
# CuTile across the local runtime versions.
def current_cuda_stream() -> int:
    """Return the current PyTorch CUDA stream in a form CuTile accepts."""
    stream = torch.cuda.current_stream()
    cuda_stream = getattr(stream, "cuda_stream", None)
    if cuda_stream is not None:
        return int(cuda_stream)
    return int(stream.stream_id)
