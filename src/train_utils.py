"""
Code for the cosine decay learning rate with linear warmup.
"""

import math
import functools
import torch


def _cosine_decay_warmup(iteration, warmup_iterations, total_iterations):
    """
    Linear warmup from 0 --> 1.0, then decay using cosine decay to 0.1
    """
    if iteration <= warmup_iterations:
        multiplier = iteration / warmup_iterations
    else:
        multiplier = (iteration - warmup_iterations) / (total_iterations - warmup_iterations)
        multiplier = max(0.1, 0.5 * (1 + math.cos(math.pi * multiplier)))
    return multiplier


def CosineAnnealingLRWarmup(optimizer, T_max, T_warmup):
    _decay_func = functools.partial(
        _cosine_decay_warmup,
        warmup_iterations=T_warmup, total_iterations=T_max
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _decay_func)
    return scheduler
