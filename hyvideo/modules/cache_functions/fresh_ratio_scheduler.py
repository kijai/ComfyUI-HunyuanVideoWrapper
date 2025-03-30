import torch
def fresh_ratio_scheduler(cache_dic, current):
    '''
    Return the fresh ratio for the current step.
    '''
    fresh_ratio = cache_dic['fresh_ratio']
    fresh_ratio_schedule = cache_dic['fresh_ratio_schedule']
    step = current['step']
    num_steps = current['num_steps']
    threshold = cache_dic['fresh_threshold']
    weight = 0.9
    if fresh_ratio_schedule == 'constant':
        return fresh_ratio
    elif fresh_ratio_schedule == 'linear':
        return fresh_ratio * (1 + weight - 2 * weight * step / num_steps)
    elif fresh_ratio_schedule == 'exp':
        #return 0.5 * (0.052 ** (step/num_steps))
        return fresh_ratio * (weight ** (step / num_steps))
    elif fresh_ratio_schedule == 'linear-mode':
        mode = (step % threshold)/threshold - 0.5
        mode_weight = 0.1
        return fresh_ratio * (1 + weight - 2 * weight * step / num_steps + mode_weight * mode)
    elif fresh_ratio_schedule == 'layerwise':
        return fresh_ratio * (1 + weight - 2 * weight * current['layer'] / 27)
    elif fresh_ratio_schedule == 'linear-layerwise':
        step_weight = -0.9 #0.9
        step_factor = 1 - step_weight + 2 * step_weight * step / num_steps
        #if current['layer'] == 2:
        #    return 1.0
        #sigmoid
        #sigmoid_weight = 0.13
        #layer_factor = 2 * torch.sigmoid(torch.tensor([sigmoid_weight * (13.5 - current['layer'])]))
        layer_weight = 0.6
        layer_factor = 1 + layer_weight - 2 * layer_weight * current['layer'] / 27

        module_weight = 1.0 #TokenCache N=8 2.5 N=6 2.5 #N=4 2.1
        module_time_weight = 0.6
        module_factor = (1 - (1-module_time_weight) * module_weight) if current['module']=='cross-attn' else (1 + module_time_weight * module_weight)
        
        return fresh_ratio * layer_factor * step_factor * module_factor

    elif fresh_ratio_schedule == 'ToCa':
        step_weight = 0.0 #0.9
        step_factor = 1 - step_weight + 2 * step_weight * step / num_steps

        layer_weight = 0.5
        layer_factor = 1 + layer_weight - 2 * layer_weight * current['layer'] / 27

        #module_weight = 1.0
        #module_time_weight = 0.6
        # this means 60*x% cross-attn computation, and 160*x% mlp computation. This is designed for cross-attn has best temporal redundancy, and mlp has worse.
        # so cross-attn compute less and mlp compute more.
        #module_factor = (1 - (1-module_time_weight) * module_weight) if current['module']=='cross-attn' else (1 + module_time_weight * module_weight)
        stream_weight = 0.6
        stream_factor = (1 - stream_weight) if current['stream']=='double_stream' else (1 + stream_weight)
        return fresh_ratio * layer_factor * step_factor * stream_factor #* module_factor

    else:
        raise ValueError("unrecognized fresh ratio schedule", fresh_ratio_schedule)
