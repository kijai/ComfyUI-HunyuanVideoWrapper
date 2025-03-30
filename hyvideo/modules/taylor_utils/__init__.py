from typing import Dict
import torch
import math

@torch.compiler.disable()
def derivative_approximation(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    """
    Compute derivative approximation
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    difference_distance = current['activated_steps'][-1] - current['activated_steps'][-2]

    updated_taylor_factors = {}
    updated_taylor_factors[0] = feature.to(cache_dic['cache_device'])

    for i in range(cache_dic['max_order']):
        if (cache_dic['cache'][-1][current['stream']][current['layer']][current['module']].get(i, None) is not None) and (current['step'] > cache_dic['first_enhance'] - 2):
            updated_factor = updated_taylor_factors[i].to(cache_dic['compute_device'])
            cached = cache_dic['cache'][-1][current['stream']][current['layer']][current['module']][i].to(cache_dic['compute_device'], non_blocking=True)
            updated_taylor_factors[i + 1] = ((updated_factor - cached) / difference_distance).to(cache_dic['cache_device'], non_blocking=True)
        else:
            break
    
    cache_dic['cache'][-1][current['stream']][current['layer']][current['module']] = updated_taylor_factors

@torch.compiler.disable()
def taylor_formula(cache_dic: Dict, current: Dict) -> torch.Tensor: 
    """
    Compute Taylor expansion error
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    x = current['step'] - current['activated_steps'][-1]
    #x = current['t'] - current['activated_times'][-1]
    output = 0
    for i in range(len(cache_dic['cache'][-1][current['stream']][current['layer']][current['module']])):
        cached = cache_dic['cache'][-1][current['stream']][current['layer']][current['module']][i]
        cached = cached.to(cache_dic['compute_device'], non_blocking=True)
        output = output + (1 / math.factorial(i)) * cached * (x ** i)
    
    return output

@torch.compiler.disable()
def taylor_cache_init(cache_dic: Dict, current: Dict):
    """
    Initialize Taylor cache, expanding storage areas for Taylor series derivatives
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    if current['step'] == 0:
        cache_dic['cache'][-1][current['stream']][current['layer']][current['module']] = {}
