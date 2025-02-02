import torch
import torch.nn.functional as F

def find_subsequence(sequence, sub_sequence):

    assert sequence.shape[0]==1
    sequence = sequence[0]
    sub_sequence = sub_sequence[0]

    sub_len = len(sub_sequence)
    indices = []
        
    windows = sequence.unfold(0, sub_len, 1)
    matches = (windows == sub_sequence).all(dim=1)
    indices = matches.nonzero().flatten().tolist()

    return indices, len(indices), sub_len

def process_single_slice(slice_str, seq_len):
    """
    Parses a single slice expression (e.g., "10", "::5", "-5:”) and returns the corresponding boolean mask.

    Args:
        slice_str (str): The slice expression.
        seq_len (int): The length of the sequence.

    Returns:
        torch.Tensor: A boolean tensor representing the mask for this slice.
    """
    slice_str = slice_str.strip()
    if ":" not in slice_str:
        # Single index
        idx = int(slice_str)
        if idx < 0:
            idx = seq_len + idx  # Handle negative indices
        mask = torch.zeros(seq_len, dtype=torch.bool)
        if 0 <= idx < seq_len:
            mask[idx] = True
        return mask

    # Handle slice notation (e.g., "start:stop:step")
    parts = slice_str.split(":")
    if len(parts) > 3:
        raise ValueError(f"Invalid slice expression: {slice_str}")

    start = int(parts[0].strip()) if parts[0].strip() else None
    stop = int(parts[1].strip()) if parts[1].strip() else None

    if len(parts) == 3:
        step = int(parts[2].strip())
    else:
        step = None

    return torch.tensor([True if i in range(seq_len)[slice(start, stop, step)] else False for i in range(seq_len)], dtype=torch.bool)

def multi_slice_to_mask(expr, seq_len):
    """
    Converts a comma-separated list of slice expressions into a boolean mask.

    Args:
        expr (str): Comma-separated slice expressions (e.g., ":10, ::5, -5:").
        seq_len (int): The length of the sequence.

    Returns:
        torch.Tensor: A boolean tensor representing the combined mask.
    """
    if not expr:
        return None  # Return None for empty string

    slices = expr.split(",")
    mask = torch.zeros(seq_len, dtype=torch.bool)

    for s in slices:
        mask |= process_single_slice(s, seq_len)

    return mask

# def multi_slice_to_mask(expr, length):
#     def process_single_slice(s):
#         s = s.replace(':', ',').replace(' ', '')
#         while ',,' in s:
#             s = s.replace(',,', ',None,')
#         if s.startswith(','):
#             s = 'None' + s
#         if s.endswith(','):
#             s = s + 'None'
#         return s
    
#     try:
#         slices = expr.split(',')
#         mask = torch.zeros(length, dtype=torch.bool)
#         if expr == "":
#             return mask
#         i = 0
#         while i < len(slices):
#             if ':' in slices[i]:
#                 slice_expr = process_single_slice(slices[i])
#                 slice_args = ast.literal_eval(f"({slice_expr})")
#                 s = slice(*slice_args)
#                 mask[s] = True
#                 i += 1
#             else:
#                 idx = ast.literal_eval(slices[i])
#                 if idx < 0:
#                     idx = length + idx
#                 if 0 <= idx < length:
#                     mask[idx] = True
#                 i += 1
                
#         return mask
#     except Exception as e:
#         raise ValueError(f"Invalid slice expression: {e}")
