import math
from copy import deepcopy
from typing import Optional

import torch
from torch import Tensor, nn


def clones(module: nn.Module, N):
    return nn.ModuleList([deepcopy(module) for _ in range(N)])


def subsequent_mask(size: int):
    attention_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attention_shape), diagonal=1).type(
        torch.uint8
    )

    return subsequent_mask == 0


def attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Optional[Tensor] = None,
    dropout: Optional[nn.Module] = None,
) -> tuple[Tensor]:
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


def make_std_mask(tgt: Tensor, pad: int):
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)

    return tgt_mask


def rate(step: int, model_size: int, factor: float, warmup: float):
    if step == 0:
        step = 1

    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )
