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



# solver.py - Example usage in other files
# import config
#
# def solve_ode(func, y0, t, **kwargs):
#     if config.TENSOR_MODE == 'concat':
#         return _solve_with_concat(func, y0, t, **kwargs)
#     else:
#         return _solve_with_loop(func, y0, t, **kwargs)

# Usage example
# import mypackage.config as config
# config.set_tensor_mode('loop')