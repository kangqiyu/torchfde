# config.py
"""Global configuration for the FDE solver package."""

import math

# Tensor handling mode: 'concat' or 'loop'
# TENSOR_MODE = 'concat'
TENSOR_MODE = 'loop'
#
"""
When the input consists of multiple tensors (e.g., tuple of tensors), this setting
determines the computational approach:

- 'concat': Concatenate tensors into a single large tensor before processing
- 'loop': Process each tensor individually in a loop

Performance Considerations:
--------------------------
'concat' (default): Better performance via vectorization and GPU utilization.
Concat/split overhead is usually outweighed by computational gains.

'loop': Lower memory usage but slower. Use when memory-constrained or 
tensors have very different sizes.
"""

"""

See _check_inputs function in utils_fde.py

CONCAT MODE:

If y0 is a tuple: flatten and concatenate all elements into a single tensor, store shapes
If y0 is a tensor: keep as-is, set tensor_input = True
Then, regardless of original type, if y0 is a tensor, wrap it in a tuple: y0 = (y0,)


LOOP MODE (non-concat):

If y0 is a tensor: set tensor_input = True
Then, if y0 is a tensor, wrap it in a tuple: y0 = (y0,)


So after _check_inputs:

y0 is ALWAYS a tuple (either original tuple, or single tensor wrapped in tuple)
tensor_input = True means original was a single tensor
shapes is set only in CONCAT mode when original was a tuple

"""


# Other global settings
# TODO: may update other setting
MEMORY = math.inf
"""
# TODO: configure the MEMORY parameter to limit history usage
#       instead of retaining the full history.
"""

def set_tensor_mode(mode):
    """Set the tensor handling mode globally.

    Args:
        mode (str): Either 'concat' or 'loop'
    """
    global TENSOR_MODE
    if mode not in ['concat', 'loop']:
        raise ValueError("Mode must be 'concat' or 'loop'")
    TENSOR_MODE = mode


def get_tensor_mode():
    """Get the current tensor handling mode."""
    return TENSOR_MODE


