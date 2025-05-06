import torch
import math
from .utils_fde import _is_tuple, _clone, _add, _multiply, _minus

def Implicit_l1(func, y0, beta, tspan, **options):
    """Use one-step Implicit_l1 method to integrate Caputo equation
        D^beta y(t) = f(t,y)
        Args:
          beta: fractional exponent in the range (0,1)
          f: callable(y,t) returning a tensor or tuple of tensors
          y0: N-D Tensor or tuple of Tensors giving the initial state vector y(t==0)
          tspan (array): The sequence of time points for which to solve for y.
            These must be equally spaced, e.g. np.arange(0,10,0.005)
            tspan[0] is the intial time corresponding to the initial state y0.
        Returns:
          y: Tensor or tuple of Tensors with the same structure as y0
        """
    N = len(tspan)
    h = (tspan[N - 1] - tspan[0]) / (N - 1)

    yn = _clone(y0)
    yn_all = [yn]
    u_h = (torch.pow(h, beta) * math.gamma(2 - beta))

    for k in range(1, N):
        tn = tspan[k]
        fhistory_k = func(tn, yn)

        # Initialize y_sum with correct structure
        if _is_tuple(y0):
            y_sum = tuple(torch.zeros_like(comp) for comp in y0)
        else:
            y_sum = 0

        for j in range(0, k - 2):
            R_k_j = torch.pow(k - j, 1 - beta) - torch.pow(k - j - 1, 1 - beta)
            y_sum = _add(y_sum, _multiply(R_k_j, _minus(yn_all[j + 1], yn_all[j])))

        u_h_f_term = _multiply(u_h, fhistory_k)
        yn = _minus(_add(yn, u_h_f_term), y_sum)

        yn_all.append(yn)

    return yn


