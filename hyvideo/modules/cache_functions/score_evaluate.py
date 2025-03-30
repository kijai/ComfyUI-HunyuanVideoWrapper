import torch
import torch.nn as nn
from .scores import attn_score, similarity_score, norm_score, k_norm_score, v_norm_score
def score_evaluate(cache_dic, tokens, current) -> torch.Tensor:
    '''
    Return the score tensor (B, N) for the given tokens.
    '''

    #if ((not current['is_force_fresh']) and (cache_dic['force_fresh'] == 'local')):
    #    # abandoned branch, if you want to explore the local force fresh strategy, this may help.
    #    force_fresh_mask = torch.as_tensor((cache_dic['cache_index'][-1][current['layer']][current['module']] >= 2 * cache_dic['fresh_threshold']), dtype = int) # 2 because the threshold is for step, not module
    #    force_len = force_fresh_mask.sum(dim=1)
    #    force_indices = force_fresh_mask.argsort(dim = -1, descending = True)[:, :force_len.min()]
    #    force_indices = force_indices[:, torch.randperm(force_indices.shape[1])]

    # Just see more explanation in the version of DiT-ToCa if needed.

    if cache_dic['cache_type'] == 'random':
        score = torch.rand(tokens.shape[0], tokens.shape[1], device=tokens.device)

    elif cache_dic['cache_type'] == 'straight':
        score = torch.ones(tokens.shape[0], tokens.shape[1]).to(tokens.device)
    
    elif cache_dic['cache_type'] == 'attention':
        # cache_dic['attn_map'][step][layer] (B, N, N), the last dimention has get softmaxed
        score = attn_score(cache_dic, current)
        #score = score + 0.0 * torch.rand_like(score, device= score.device)
    
    elif cache_dic['cache_type'] == 'similarity':
        score = similarity_score(cache_dic, current, tokens)

    elif cache_dic['cache_type'] == 'norm':
        score = norm_score(cache_dic, current, tokens)

    elif cache_dic['cache_type'] == 'k-norm':
        score = k_norm_score(cache_dic, current)

    elif cache_dic['cache_type'] == 'v-norm':
        score = v_norm_score(cache_dic, current)

    elif cache_dic['cache_type'] == 'compress':
        score1 = torch.rand(int(tokens.shape[0]*0.5), tokens.shape[1])
        score1 = torch.cat([score1, score1], dim=0).to(tokens.device)
        score2 = cache_dic['attn_map'][-1][current['layer']].sum(dim=1)#.mean(dim=0) # (B, N)
        # normalize
        score2 = score2 / score2.max(dim=1, keepdim=True)[0]
        score = 0.5 * score1 + 0.5 * score2
    
    # abandoned the branch, if you want to explore the local force fresh strategy, this may help.
    #if ((not current['is_force_fresh']) and (cache_dic['force_fresh'] == 'local')): # current['is_force_fresh'] is False, cause when it is True, no cut and fresh are needed
    #        #print(torch.ones_like(force_indices, dtype=float, device=force_indices.device).dtype)
    #    score.scatter_(dim=1, index=force_indices, src=torch.ones_like(force_indices, dtype=torch.float32, 
    #                                                                       device=force_indices.device))
    
    ###if (True and (cache_dic['force_fresh'] == 'global')):
    ###    soft_step_score = cache_dic['cache_index'][-1][current['layer']][current['module']].float() / (cache_dic['fresh_threshold'])
    ###    #soft_layer_score = cache_dic['cache_index']['layer_index'][current['module']].float() / (27)
    ###    score = score + cache_dic['soft_fresh_weight'] * soft_step_score #+ 0.1 *soft_layer_score
    
    return score.to(tokens.device)