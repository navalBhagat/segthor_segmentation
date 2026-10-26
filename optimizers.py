# Support 4 optimizers: Adam, SGD, RMSprop, AdamW
from typing import Any

import torch

def get_optimizer(net: torch.nn.Module, optimizer_name: str = 'adam', lr: float = 0.001, kwargs: dict[str, Any] | None = None) -> torch.optim.Optimizer:
    optimizer_kwargs: dict[str, Any] = {'lr': lr}
    optimizer_kwargs.update(kwargs or {})

    if optimizer_name == 'adam':
        optimizer_kwargs.setdefault('betas', (0.9, 0.999))
        return torch.optim.Adam(net.parameters(), **optimizer_kwargs)
    elif optimizer_name == 'sgd':
        optimizer_kwargs.setdefault('momentum', 0.9)
        return torch.optim.SGD(net.parameters(), **optimizer_kwargs)
    elif optimizer_name == 'rmsprop':
        optimizer_kwargs.setdefault('alpha', 0.99)
        return torch.optim.RMSprop(net.parameters(), **optimizer_kwargs)
    elif optimizer_name == 'adamw':
        optimizer_kwargs.setdefault('betas', (0.9, 0.999))
        return torch.optim.AdamW(net.parameters(), **optimizer_kwargs)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")