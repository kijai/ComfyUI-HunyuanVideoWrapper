# Besides, re-arrange the attention module
from torch.jit import Final
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union
#from xformers.ops.fmha.attn_bias import BlockDiagonalMask
def cached_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    #attn_bias: Optional[Union[torch.Tensor, BlockDiagonalMask]] = None,
    attn_bias,
    p: float = 0.0,
    scale: Optional[float] = None
) -> torch.Tensor:
    scale = 1.0 / query.shape[-1] ** 0.5
    query = query * scale
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    attn = query @ key.transpose(-2, -1)
    if attn_bias is not None:
        attn_bias = attn_bias.materialize(shape= attn.shape, dtype= attn.dtype, device= attn.device)
        attn = attn + attn_bias
    #out_map = attn
    attn_map = attn.softmax(-1)
    attn = F.dropout(attn_map, p)
    attn = attn @ value

    return attn.transpose(1, 2).contiguous(), attn_map.mean(dim=1)