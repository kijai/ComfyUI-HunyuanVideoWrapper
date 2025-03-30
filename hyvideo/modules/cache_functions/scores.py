import torch
import torch.nn as nn
import torch.nn.functional as F

def attn_score(cache_dic, current):
    #self_attn_score = 1- cache_dic['attn_map'][-1][current['layer']].diagonal(dim1=1, dim2=2)
    #self_attn_score = F.normalize(self_attn_score, dim=1, p=2)
    #attention_score = F.normalize(cache_dic['attn_map'][-1][current['layer']].sum(dim=1), dim=1, p=2)
    #cross_attn_map = F.threshold(cache_dic['cross_attn_map'][-1][current['layer']],threshold=0.0, value=0.0)
    #cross_attention_score = F.normalize(cross_attn_map.sum(dim=-1), dim=-1, p=2)

    # Note: It is important to give a same selection method for cfg and no cfg.
    # Because the influence of **Cross-Attention** in text-contidional models makes cfg and no cfg a BIG difference.

    # Same selection for cfg and no cfg
    #cond_cmap, uncond_cmap = torch.split(cache_dic['attn_map'][-1][current['layer']], len(cache_dic['cross_attn_map'][-1][current['layer']]) // 2, dim=0)
    #cond_weight = 0.5
    #cmap = cond_weight * cond_cmap + (1 - cond_weight) * uncond_cmap

    ## Entropy score
    #cross_attention_entropy = -torch.sum(cmap * torch.log(cmap + 1e-7), dim=-1)
    #cross_attention_score   = F.normalize(1 + cross_attention_entropy, dim=1, p=2) # Note here "1" does not influence the sorted sequence, but provie stability.
    #score = cross_attention_score.repeat(2, 1)
    if current['stream'] == 'double_stream':
        score = F.normalize(cache_dic['attn_map'][-1][current['stream']][current['layer']][current['module']], dim=-1, p=2)
    elif current['stream'] == 'single_stream':
        score = F.normalize(cache_dic['attn_map'][-1][current['stream']][current['layer']]['total'], dim=-1, p=2)

    # You can try conbining the self_attention_score (s1) and cross_attention_score (s2) as the final score, there exists a balance.
    #cross_weight = 0.0
    #score =  (1-cross_weight) * attention_score + cross_weight * cross_attention_score
    return score

def similarity_score(cache_dic, current, tokens):
    cosine_sim = F.cosine_similarity(tokens, cache_dic['cache'][-1][current['layer']][current['module']], dim=-1)

    return F.normalize(1- cosine_sim, dim=-1, p=2)

def norm_score(cache_dic, current, tokens):
    norm = tokens.norm(dim=-1, p=2)
    return F.normalize(norm, dim=-1, p=2)

def kv_norm_score(cache_dic, current):
    # (B, N, num_heads)
    #cond_k_norm, uncond_k_norm = torch.split(cache_dic['cache'][-1][current['layer']]['k_norm'], len(cache_dic['cache'][-1][current['layer']]['k_norm']) // 2, dim=0)
    cond_v_norm, uncond_v_norm = torch.split(cache_dic['cache'][-1][current['layer']]['v_norm'], len(cache_dic['cache'][-1][current['layer']]['v_norm']) // 2, dim=0)
    cond_weight = 0.5
    #k_norm = cond_weight * cond_k_norm + (1 - cond_weight) * uncond_k_norm
    v_norm = cond_weight * cond_v_norm + (1 - cond_weight) * uncond_v_norm
    kv_norm = 1 -v_norm

    ## 计算 (B/2, N) 张量在 N 维度上的每个元素与均值的绝对值差
    #kv_norm_mean = kv_norm.mean(dim=-2, keepdim=True)
    #kv_norm_diff = torch.abs(kv_norm - kv_norm_mean)
    
    return F.normalize(kv_norm.sum(dim=-1), p=2).repeat(2, 1)

def k_norm_score(cache_dic, current):
    # (B, N)

    if current['stream'] == 'double_stream':
        score = F.normalize(cache_dic['k-norm'][-1][current['stream']][current['layer']][current['module']], dim=-1, p=2)
    elif current['stream'] == 'single_stream':
        score = F.normalize(cache_dic['k-norm'][-1][current['stream']][current['layer']]['total'], dim=-1, p=2)

    return score

def v_norm_score(cache_dic, current):
    # (B, N)

    if current['stream'] == 'double_stream':
        score = F.normalize(cache_dic['v-norm'][-1][current['stream']][current['layer']][current['module']], dim=-1, p=2)
    elif current['stream'] == 'single_stream':
        score = F.normalize(cache_dic['v-norm'][-1][current['stream']][current['layer']]['total'], dim=-1, p=2)

    return score

