import torch
import math
from .utils_fde import _is_tuple, _clone, _add, _multiply

def fractional_pow(base, exponent):
    eps = 1e-4
    return torch.pow(base, exponent)

def Predictor(func, y0, beta, tspan, **options):
    """Use one-step Adams-Bashforth (Euler) method to integrate Caputo equation
        D^beta y(t) = f(t,y)
        Args:
          beta: fractional exponent in the range (0,1)
          f: callable(y,t) returning a numpy array of shape (d,)
             Vector-valued function to define the right hand side of the system
          y0: N-D Tensor or tuple of Tensors giving the initial state vector y(t==0)
          tspan (array): The sequence of time points for which to solve for y.
            These must be equally spaced, e.g. np.arange(0,10,0.005)
            tspan[0] is the intial time corresponding to the initial state y0.
        Returns:
          y: Tensor or tuple of Tensors with the same structure as y0
             With the initial value y0 in the first row
        """
    N = len(tspan)
    h = (tspan[-1] - tspan[0]) / (N - 1)
    gamma_beta = 1 / math.gamma(beta)
    fhistory = []
    h_beta_over_beta = torch.pow(h, beta) / beta
    # Get device from y0 (handle both tensor and tuple cases)
    if _is_tuple(y0):
        device = y0[0].device
    else:
        device = y0.device

    yn = _clone(y0)

    for k in range(N-1):
        tn = tspan[k]
        f_k = func(tn, yn)
        fhistory.append(f_k)

        # can apply short memory here
        if 'memory' not in options:
            memory = k
        else:
            memory = options['memory']
        memory_k = max(0, k - memory)

        j_vals = torch.arange(0, k + 1, dtype=torch.float32, device=device).unsqueeze(1)
        b_j_k_1 = h_beta_over_beta * (torch.pow(k + 1 - j_vals, beta) - torch.pow(k - j_vals, beta))

        # Initialize accumulator with correct structure (tensor or tuple)
        if _is_tuple(fhistory[memory_k]):
            b_all_k = tuple(torch.zeros_like(f_i) for f_i in fhistory[memory_k])
        else:
            b_all_k = torch.zeros_like(fhistory[memory_k])

        # Loop through the range and accumulate results
        for i in range(memory_k, k + 1):
            b_all_k = _add(b_all_k, _multiply(b_j_k_1[i], fhistory[i]))


        # Final update step
        weight_term = _multiply(gamma_beta, b_all_k)
        yn = _add(y0, weight_term)

    # release memory
    del fhistory
    del b_j_k_1
    return yn

def Predictor_Corrector(func, y0, beta, tspan,**options):
    """Use one-step Adams-Bashforth (Euler) method to integrate Caputo equation
        D^beta y(t) = f(t,y)

        Raises:
          FODEValueError
        See also:
          K. Diethelm et al. (2004) Detailed error analysis for a fractional Adams
             method
          C. Li and F. Zeng (2012) Finite Difference Methods for Fractional
             Differential Equations
        """
    N = len(tspan)
    h = (tspan[-1] - tspan[0]) / (N - 1)
    gamma_beta = 1 / math.gamma(beta)
    a_item = torch.pow(h, beta) / (beta * (beta + 1))
    fhistory = []

    # Get device from y0 (handle both tensor and tuple cases)
    if _is_tuple(y0):
        device = y0[0].device
    else:
        device = y0.device

    yn = _clone(y0)

    for k in range(N-1):
        tn = tspan[k]
        f_k = func(tn, yn)
        fhistory.append(f_k)

        # can apply short memory here
        if 'memory' not in options:
            memory = k
        else:
            memory = options['memory']
        memory_k = max(0, k - memory)

        j_vals = torch.arange(0, k + 1, dtype=torch.float32, device=device).unsqueeze(1)
        b_j_k_1 = (fractional_pow(h, beta) / beta) * (
                fractional_pow(k + 1 - j_vals, beta) - fractional_pow(k - j_vals, beta))

        # Initialize accumulator with correct structure (tensor or tuple)
        if _is_tuple(fhistory[memory_k]):
            b_all_k = tuple(torch.zeros_like(f_i) for f_i in fhistory[memory_k])
        else:
            b_all_k = torch.zeros_like(fhistory[memory_k])

        # Loop through the range and accumulate results
        for i in range(memory_k, k + 1):
            b_all_k = _add(b_all_k, _multiply(b_j_k_1[i], fhistory[i]))

        # Final update step
        weight_term = _multiply(gamma_beta, b_all_k)
        yn_ = _add(y0, weight_term)



        a_j_k_1 = a_item * torch.ones((k + 2, 1), dtype=torch.float32, device=device)
        a_j_k_1[0] = a_item * (torch.pow(k, beta + 1) - (k - beta) * torch.pow(k + 1, beta))
        for j in range(1, k + 1):
            a_j_k_1[j] = a_item * (
                        torch.pow(k + 2 - j, beta + 1) + torch.pow(k - j, beta + 1) - 2 * torch.pow(k + 1 - j,
                                                                                                    beta + 1))
        # corrector
        if 'corrector_step' not in options:
            corrector_step = 1
        else:
            corrector_step = options['corrector_step']

        if _is_tuple(fhistory[memory_k]):
            a_all_k = tuple(torch.zeros_like(f_i) for f_i in fhistory[memory_k])
        else:
            a_all_k = torch.zeros_like(fhistory[memory_k])

        for i in range(memory_k, k + 1):
            a_all_k = _add(a_all_k, _multiply(a_j_k_1[i], fhistory[i]))

        for _ in range(corrector_step):
            a_all_k_final = _add(a_all_k, _multiply(a_j_k_1[k+1], func(tn+h, yn_)))
            # multiple updates for corrector
            weight_term = _multiply(gamma_beta, a_all_k_final)
            yn_ = _add(y0, weight_term)

        yn = yn_

        # release memory
        del fhistory
        del b_j_k_1

        return yn






