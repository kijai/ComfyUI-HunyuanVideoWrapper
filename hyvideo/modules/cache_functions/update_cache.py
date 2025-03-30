import torch
def update_cache(fresh_indices, fresh_tokens, cache_dic, current, fresh_attn_map=None):
    '''
    Update the cache with the fresh tokens.
    '''
    step = current['step']
    layer = current['layer']
    module = current['module']
    # Update the cached tokens at the positions


    indices = fresh_indices
    
    cache_dic['cache'][-1][current['stream']][current['layer']][current['module']][0].scatter_(dim=1, index=indices.unsqueeze(-1).expand(-1, -1, fresh_tokens.shape[-1]), src=fresh_tokens)
    
    

        
        