import numpy as np
import torch

def get_torch_dtype(precision):
    """Maps a PyTorch Lightning `trainer.precision` string to the corresponding torch dtype."""
    if precision in ('32-true', '32'):
        return torch.float32
    elif precision in ('16-true', '16-mixed', '16'):
        return torch.float16
    elif precision in ('bf16-true', 'bf16-mixed', 'bf16'):
        return torch.bfloat16
    else:
        raise ValueError(f"Unsupported precision: {precision}")


def from_numpy(data):
    """Recursively transform numpy.ndarray to torch.Tensor.
    """
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = from_numpy(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [from_numpy(x) for x in data]
    if isinstance(data, np.ndarray):
        """Pytorch now has bool type."""
        data = torch.from_numpy(data)
    return data