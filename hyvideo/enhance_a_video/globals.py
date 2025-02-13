import torch

NUM_FRAMES = None
FETA_WEIGHT = None
ENABLE_FETA_SINGLE = False
ENABLE_FETA_DOUBLE = False

@torch.compiler.disable()
def set_num_frames(num_frames: int):
    global NUM_FRAMES
    NUM_FRAMES = num_frames

@torch.compiler.disable()
def get_num_frames() -> int:
    return NUM_FRAMES


def enable_enhance(single, double):
    global ENABLE_FETA_SINGLE, ENABLE_FETA_DOUBLE
    ENABLE_FETA_SINGLE = single
    ENABLE_FETA_DOUBLE = double

def disable_enhance():
    global ENABLE_FETA_SINGLE, ENABLE_FETA_DOUBLE
    ENABLE_FETA_SINGLE = False
    ENABLE_FETA_DOUBLE = False

@torch.compiler.disable()
def is_enhance_enabled_single() -> bool:
    return ENABLE_FETA_SINGLE

@torch.compiler.disable()
def is_enhance_enabled_double() -> bool:
    return ENABLE_FETA_DOUBLE

@torch.compiler.disable()
def set_enhance_weight(feta_weight: float):
    global FETA_WEIGHT
    FETA_WEIGHT = feta_weight

@torch.compiler.disable()
def get_enhance_weight() -> float:
    return FETA_WEIGHT
