import torch

def force_init(cache_dic, current, tokens):
    '''
    Initialization for Force Activation step.
    '''
    cache_dic['cache_index'][-1][current['layer']][current['module']] = torch.zeros(tokens.shape[0], tokens.shape[1], dtype=torch.int, device=tokens.device)

    #if current['layer'] == 0:
    #    cache_dic['cache_index']['layer_index'][current['module']] = torch.zeros(tokens.shape[0], tokens.shape[1], dtype=torch.int, device=tokens.device)