import torch
from typing import Dict

def support_set_selection(x: torch.Tensor, fresh_ratio: float, base_ratio: float, current: Dict, cache_dic: Dict) -> torch.Tensor:
    
    #selection_start = 0
    #
    #if current['stream'] == 'single_stream':
    #    # only select from the img tokens
    #    x = x[:, cache_dic['txt_shape'] :]
    #    selection_start = cache_dic['txt_shape']

    B, N, H = x.shape
    num_total = int(fresh_ratio * N)         # 最终每个 batch 选取的 token 数
    base_count = int(base_ratio * num_total)  # 随机选取的 token 数
    #base_count = 1
    add_count = num_total - base_count  # 需要从候选集中选取的 token 数

    # 1. 随机选取 (B, base_count) 个 token
    random_indices = torch.randperm(N, device=x.device)
    base_indices = random_indices[:base_count]
    other_indices = random_indices[base_count:]

    base_tokens = x.gather(dim=1, index=base_indices.unsqueeze(-1).expand(B, -1, H))
    #other_tokens = x.gather(dim=1, index=other_indices.unsqueeze(-1).expand(-1, -1, H))

    # 2. 计算余下 token 与已选 token 的相似度
    
    # normaize
    base_tokens = base_tokens / base_tokens.norm(dim=-1, keepdim=True)
    #other_tokens = other_tokens / other_tokens.norm(dim=-1, keepdim=True)
    x_norm = x / x.norm(dim=-1, keepdim=True)

    # 计算余下 token 与已选 token 的相似度
    similarity = torch.einsum('bnd,bmd->bnm', base_tokens, x_norm)

    # 计算每列最小值
    min_similarity = similarity.min(dim=1).values
    #min_similarity = similarity.max(dim=1).values

    # 3. 选取相似度最小的 token
    _, min_indices = min_similarity.topk(add_count, largest=False)
    #_, min_indices = min_similarity.topk(add_count, largest=True)

    # 4. 合并 base_indices 和 min_indices
    #indices = torch.cat([base_indices, other_indices[min_indices]], dim=-1)
    indices = torch.cat([base_indices.expand(B, -1), min_indices], dim=-1) #+ selection_start

    return indices


    