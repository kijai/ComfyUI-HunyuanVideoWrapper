from .force_scheduler import force_scheduler
def global_force_fresh(cache_dic, current):
    '''
    Return whether to force fresh tokens globally.
    '''
    first_step = (current['step'] == 0)
    second_step = (current['step'] == 1)
    force_fresh = cache_dic['force_fresh']
    if not first_step:
        fresh_threshold = cache_dic['cal_threshold']
    else:
        fresh_threshold = cache_dic['fresh_threshold']

    if force_fresh == 'global':
        return (first_step or (current['step']% fresh_threshold == 0))
    elif force_fresh == 'local':
        return first_step
    elif force_fresh == 'none':
        return first_step
    else:
        raise ValueError("unrecognized force fresh strategy", force_fresh)