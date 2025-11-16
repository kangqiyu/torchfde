import torch
import math
from .utils_fde import _is_tuple, _clone, _add, _multiply, _minus
from . import config

def GLmethod(func, y0, beta, tspan, **options):
    """Use GL method to integrate Riemann-Liouville equation
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

    # Get device from y0 (handle both tensor and tuple cases)
    if _is_tuple(y0):
        device = y0[0].device
    else:
        device = y0.device

    c = torch.zeros(N + 1, dtype=torch.float64, device=device)
    c[0] = 1
    for j in range(1, N + 1):
        c[j] = (1 - (1 + beta) / j) * c[j - 1]

    yn = _clone(y0)
    y_history = [yn]

    for k in range(1, N):
        tn = tspan[k]

        # Initialize right term with correct structure
        if _is_tuple(y0):
            right = tuple(torch.zeros_like(comp) for comp in y0)
        else:
            right = 0
        
        for j in range(1, k + 1):
            right = _add(right, _multiply(c[j], y_history[k - j]))

        # Calculate f(tn, yn) * h^beta term
        f_term = func(tn, yn)
        h_power = torch.pow(h, beta)

        f_h_term = _multiply(h_power, f_term)
        # Subtract right from f_h_term
        yn = _minus(f_h_term, right)
        y_history.append(yn)

    del y_history
    return yn

def RLcoeffs(index_k, index_j, alpha):
    """Calculates coefficients for the RL differintegral operator.

    see Baleanu, D., Diethelm, K., Scalas, E., and Trujillo, J.J. (2012). Fractional
        Calculus: Models and Numerical Methods. World Scientific.
    """

    if index_j == 0:
        return ((index_k - 1) ** (1 - alpha) - (index_k + alpha - 1) * index_k ** -alpha)
    elif index_j == index_k:
        return 1
    else:
        return ((index_k - index_j + 1) ** (1 - alpha) + (index_k - index_j - 1) ** (1 - alpha) - 2 * (
                    index_k - index_j) ** (1 - alpha))



def Product_Trap(func, y0, beta, tspan, **options):
    """Use Product Trapezoidal method to integrate fractional equation
    Args:
      beta: fractional exponent in the range (0,1)
      f: callable(y,t) returning a tensor or tuple of tensors
      y0: N-D Tensor or tuple of Tensors giving the initial state vector y(t==0)
      tspan (array): The sequence of time points for which to solve for y.
    Returns:
      y: Tensor or tuple of Tensors with the same structure as y0
    """
    N = len(tspan)
    h = (tspan[N - 1] - tspan[0]) / (N - 1)

    h_power = torch.pow(h, beta)
    gamma_factor = math.gamma(2 - beta)

    yn = _clone(y0)
    y_history = [yn]

    for k in range(1, N):
        tn = tspan[k]

        # Initialize right term with correct structure
        if _is_tuple(y0):
            right = tuple(torch.zeros_like(comp) for comp in y0)
        else:
            right = 0

        for j in range(0, k):
            coeff = RLcoeffs(k, j, beta)
            # Handle tuple case
            right = _add(right, _multiply(coeff, y_history[j]))

        # Calculate gamma * f(tn, yn) * h^beta term
        f_term = func(tn, yn)

        f_h_term = _multiply(h_power * gamma_factor, f_term)
        # Subtract right from f_h_term
        yn = _minus(f_h_term, right)
        y_history.append(yn)

    del y_history
    return yn
