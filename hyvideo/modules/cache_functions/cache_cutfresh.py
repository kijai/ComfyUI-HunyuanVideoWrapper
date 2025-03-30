from .fresh_ratio_scheduler import fresh_ratio_scheduler
from .score_evaluate import score_evaluate
#from .token_merge import token_merge
from .support_set_selection import support_set_selection
import torch
def cache_cutfresh(cache_dic, tokens, current):
    '''
    Cut fresh tokens from the input tokens and update the cache counter.
    
    cache_dic: dict, the cache dictionary containing cache(main extra memory cost), indices and some other information.
    tokens: torch.Tensor, the input tokens to be cut.
    current: dict, the current step, layer, and module information. Particularly convenient for debugging.
    '''
    step = current['step']
    layer = current['layer']
    stream = current['stream']
    module = current['module']
    
    fresh_ratio = fresh_ratio_scheduler(cache_dic, current)
    fresh_ratio = torch.clamp(torch.tensor(fresh_ratio, device = tokens.device), min=0, max=1)
    
    # Generate the index tensor for fresh tokens
    score = score_evaluate(cache_dic, tokens, current) # s1, s2, s3 mentioned in the paper
    #score = local_selection_with_bonus(score, 0.4, 4) # Uniform Spatial Distribution s4 mentioned in the paper
    indices = score.argsort(dim=-1, descending=True)
    topk = int(fresh_ratio * score.shape[1])
    fresh_indices = indices[:, :topk]
    stale_indices = indices[:, topk:]

    #fresh_indices = support_set_selection(tokens, fresh_ratio, 0.4, current, cache_dic) # (B, fresh_ratio * N) # 0.4

    # (B, fresh_ratio *N)

    # Updating the Cache Frequency Score s3 mentioned in the paper
    # stale tokens index + 1 in each ***module***, fresh tokens index = 0
    ###cache_dic['cache_index'][-1][layer][module] += 1
    ###cache_dic['cache_index'][-1][layer][module].scatter_(dim=1, index=fresh_indices, 
    ###                                                                src = torch.zeros_like(fresh_indices, dtype=torch.int, device=fresh_indices.device))
    #cache_dic['cache_index']['layer_index'][module] += 1
    #cache_dic['cache_index']['layer_index'][module].scatter_(dim=1, index=fresh_indices, 
    #                                                                src = torch.zeros_like(fresh_indices, dtype=torch.int, device=fresh_indices.device))
    
    fresh_indices_expand = fresh_indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])

    fresh_tokens = torch.gather(input = tokens, dim = 1, index = fresh_indices_expand)
    return fresh_indices, fresh_tokens
    
def local_selection_with_bonus(score, bonus_ratio, grid_size=2):
    batch_size, num_tokens = score.shape
    image_size = int(num_tokens ** 0.5)
    block_size = grid_size * grid_size
    
    assert num_tokens % block_size == 0, "The number of tokens must be divisible by the block size."
    
    # Step 1: Reshape score to group it by blocks
    score_reshaped = score.view(batch_size, image_size // grid_size, grid_size, image_size // grid_size, grid_size)
    score_reshaped = score_reshaped.permute(0, 1, 3, 2, 4).contiguous()
    score_reshaped = score_reshaped.view(batch_size, -1, block_size)  # [batch_size, num_blocks, block_size]
    
    # Step 2: Find the max token in each block
    max_scores, max_indices = score_reshaped.max(dim=-1, keepdim=True)  # [batch_size, num_blocks, 1]
    
    # Step 3: Create a mask to identify max score tokens
    mask = torch.zeros_like(score_reshaped)
    mask.scatter_(-1, max_indices, 1)  # Set mask to 1 at the max indices
    
    # Step 4: Apply the bonus only to the max score tokens
    score_reshaped = score_reshaped + (mask * max_scores * bonus_ratio)  # Apply bonus only to max tokens
    
    # Step 5: Reshape the score back to its original shape
    score_modified = score_reshaped.view(batch_size, image_size // grid_size, image_size // grid_size, grid_size, grid_size)
    score_modified = score_modified.permute(0, 1, 3, 2, 4).contiguous()
    score_modified = score_modified.view(batch_size, num_tokens)
    
    return score_modified