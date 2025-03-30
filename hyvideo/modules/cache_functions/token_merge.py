import torch
def token_merge(cache_dic, tokens, current, fresh_indices, stale_indices):
    '''
    An abandoned branch in exploring if token merge helps. The answer is no, at least no for training-free strategy.
    '''
    if (current['layer'] % 1 == 0):
        fresh_tokens = torch.gather(input = tokens, dim = 1, index = fresh_indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))
        stale_tokens = torch.gather(input = tokens, dim = 1, index = stale_indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))
        method = 'similarity'
        if method == 'distance':
            descending = False
            distance = torch.cdist(stale_tokens, fresh_tokens, p=1)
            stale_fresh_dist, stale_fresh_indices_allstale = torch.min(distance, dim=2)
        elif method == 'similarity':
            descending = True
            fresh_tokens = torch.nn.functional.normalize(fresh_tokens, p=2, dim=-1)
            stale_tokens = torch.nn.functional.normalize(stale_tokens, p=2, dim=-1)
            similarity = stale_tokens @ fresh_tokens.transpose(1, 2)
            stale_fresh_dist, stale_fresh_indices_allstale = torch.max(similarity, dim=2)
        

        saved_topk_stale = int((stale_fresh_dist > 0.995).sum(dim=1).min())
        merged_stale_sequence = torch.sort(stale_fresh_dist, dim=1, descending=descending)[1][:,:saved_topk_stale]
        stale_fresh_indices = stale_fresh_indices_allstale.gather(1, merged_stale_sequence)
        merged_stale_sequence = stale_indices.gather(1, merged_stale_sequence)
        merged_stale_fresh_indices = fresh_indices.gather(1, stale_fresh_indices)
        cache_dic['merged_stale_fresh_indices'] = merged_stale_fresh_indices
        cache_dic['merged_stale_sequence'] = merged_stale_sequence 
