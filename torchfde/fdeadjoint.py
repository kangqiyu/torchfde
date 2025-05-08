import torch
import math
import torch.nn as nn
from .utils_fde import _flatten, _flatten_convert_none_to_zeros,_check_inputs, _flat_to_shape
from .utils_fde import _is_tuple, _clone, _add, _multiply, _minus, ReversedListView
# from . import fdeint
# from .explicit_solver import Predictor,Predictor_Corrector
# from .implicit_solver import Implicit_l1
from .riemann_liouville_solver import RLcoeffs
# import pdb


class FDEAdjointMethod(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args):
        assert len(args) >= 7, 'Internal error: all arguments required.'
        y0, func, beta, tspan, flat_params, method, options = \
            args[:-6], args[-6], args[-5], args[-4], args[-3], args[-2], args[-1]
        y0 = tuple(y0)
        ctx.func = func
        ctx.beta = beta
        ctx.method = method

        with torch.no_grad():
            ans, yhistory = SOLVERS[method](func=func, y0=y0, beta=beta, tspan=tspan,**options)

        if any(t.requires_grad for t in [*y0, *flat_params]):
            ctx.save_for_backward(tspan, flat_params, *ans)
            ctx.yhistory = yhistory
        else:
            del yhistory

        return ans

    @staticmethod
    def backward(ctx, *grad_output):
        tspan, flat_params, *ans = ctx.saved_tensors
        yhistory = ctx.yhistory
        ans = tuple(ans)
        # yhistory = yhistory_#.clone().detach().requires_grad_(True)

        func = ctx.func
        beta = ctx.beta
        method = ctx.method
        f_params = tuple(func.parameters())
        n_tensors = len(ans)

        def augmented_dynamics(t, y_aug):
            y, adj_y, _ = y_aug

            with torch.set_grad_enabled(True):
                y = tuple(y_.detach().requires_grad_(True) for y_ in y)
                func_eval = func(t, y)
                vjp_y_and_params = torch.autograd.grad(
                    func_eval, y + f_params,
                    tuple(adj_y_ for adj_y_ in adj_y), allow_unused=True, retain_graph=True
                )

            vjp_y = vjp_y_and_params[:n_tensors]
            vjp_params = vjp_y_and_params[n_tensors:]

            # autograd.grad returns None if no gradient, set to zero.
            vjp_y = tuple(torch.zeros_like(y_) if vjp_y_ is None else vjp_y_ for vjp_y_, y_ in zip(vjp_y, y))
            vjp_params = _flatten_convert_none_to_zeros(vjp_params, f_params)

            if len(f_params) == 0:
                vjp_params = torch.tensor(0.).to(vjp_y[0])
            return (func_eval, vjp_y, vjp_params)  ## return (tuple, tuple, vector)

        tspan_flip = tspan.flip(0)

        if yhistory is not None:
            yhistory_flip = ReversedListView(yhistory)
        else:
            yhistory_flip = None

        with torch.no_grad():
            adj_y = grad_output
            adj_params = torch.zeros_like(flat_params)

            if adj_params.numel() == 0:
                adj_params = torch.tensor(0.).to(adj_y[0])

            aug_y0 = (ans, adj_y, adj_params)

            if method == 'predictor-f':
                backwardmethod = BackwardMixedOrderPredictor_f
            elif method == 'predictor-o':
                backwardmethod = BackwardMixedOrde_o
            elif method == 'gl-f':
                backwardmethod = BackwardMixedOrderGLmethod_f
            elif method == 'gl-o':
                backwardmethod = BackwardMixedOrde_o#mixOrderGLmethod_o
            elif method == 'trap-f':
                backwardmethod = BackwardMixedOrderTrap_f
            elif method == 'trap-o':
                backwardmethod = BackwardMixedOrde_o

            adj_y, adj_params = backwardmethod(
                augmented_dynamics, aug_y0, beta,
                tspan_flip, yhistory_flip)

        del yhistory_flip, ctx.yhistory

        return (*adj_y, None, None, None, adj_params, None, None)


def fdeint_adjoint(func,y0,beta,t,step_size,method,options=None):

    # We need this in order to access the variables inside this module,
    # since we have no other way of getting variables along the execution path.
    if not isinstance(func, nn.Module):
        raise ValueError('func is required to be an instance of nn.Module.')

    tensor_input = False
    if torch.is_tensor(y0):
        class TupleFunc(nn.Module):

            def __init__(self, base_func):
                super(TupleFunc, self).__init__()
                self.base_func = base_func

            def forward(self, t, y):
                return (self.base_func(t, y[0]),)

        tensor_input = True
        y0 = (y0,)
        func = TupleFunc(func)

    shapes, tensor_input, func, y0, tspan, method, beta = _check_inputs(func, y0, t, step_size, method, beta, SOLVERS)

    if options is None:
        options = {}

    flat_params = _flatten(func.parameters())
    solution = FDEAdjointMethod.apply(*y0, func, beta, tspan, flat_params, method, options)

    assert tensor_input, 'tensor_input should be true since we flatten the input to be tensor in this version'
    solution = solution[0]
    if shapes is not None:
        solution = _flat_to_shape(solution, (), shapes)

    return solution


def fractional_pow(base, exponent):
    eps = 1e-4
    return torch.pow(base, exponent)

def BackwardMixedOrderPredictor_f(func, y_aug, beta, tspan, yhistory, **options):
# mixed order predictor with beta and 1.
    with torch.no_grad():
        N = len(tspan)
        # print("N: ", N)
        h = (tspan[-1] - tspan[0]) / (N - 1)
        h = torch.abs(h)
        # print("h: ", h)
        gamma_beta =  1 / math.gamma(beta)
        fadj_history = []

        if True:#yhistory is None:
            fy_history = []

        y0, adj_y0, adj_params0 = y_aug ### we will use yhistory rather than compute y again

        if _is_tuple(y0):
            device = y0[0].device
        else:
            device = y0.device

        adj_y = _clone(adj_y0)
        adj_params = _clone(adj_params0)
        y = _clone(y0)

        for k in range(N-1):
            tn = tspan[k]
            j_vals = torch.arange(0, k + 1, dtype=torch.float32, device=device).unsqueeze(1)
            b_j_k_1 = (fractional_pow(h, beta) / beta) * (
                        fractional_pow(k + 1 - j_vals, beta) - fractional_pow(k - j_vals, beta))


            fy_k, fadj_k, jp_params_k = func(tn, (y, adj_y, adj_params))
            fadj_history.append(fadj_k)

            if yhistory is None:
                fy_history.append(fy_k)
            # can apply short memory here
            if True:#'memory' not in options:
                memory = k
            else:
                memory = options['memory']
            memory_k = max(0, k - memory)

            if _is_tuple(fadj_history[memory_k]):
                b_all_k = tuple(torch.zeros_like(f_i) for f_i in fadj_history[memory_k])
            else:
                b_all_k = torch.zeros_like(fadj_history[memory_k])

            # Loop through the range and accumulate results
            for i in range(memory_k, k + 1):
                b_all_k = _add(b_all_k, _multiply(b_j_k_1[i], fadj_history[i]))


            # Final update step
            weight_term = _multiply(gamma_beta, b_all_k)
            adj_y = _add(adj_y0, weight_term)

            if yhistory is not None and k<N-1:
                y = yhistory[k+1]
            elif yhistory is None:

                if _is_tuple(fy_history[memory_k]):
                    b_all_k = tuple(torch.zeros_like(f_i) for f_i in fy_history[memory_k])
                else:
                    b_all_k = torch.zeros_like(fy_history[memory_k])

                # Loop through the range and accumulate results
                for i in range(memory_k, k + 1):
                    b_all_k = _add(b_all_k, _multiply(b_j_k_1[i], fy_history[i]))

                weight_term = _multiply(gamma_beta, b_all_k)
                y = _add(y0, weight_term)

            adj_params = adj_params + h * jp_params_k

    # release memory
    del fadj_history, yhistory, fy_history
    del b_j_k_1
    return adj_y, adj_params


def ForwardPredictor_w_History(func, y0, beta, tspan, **options):
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
    with torch.no_grad():
        N = len(tspan)
        h = (tspan[-1] - tspan[0]) / (N - 1)
        h = torch.abs(h)
        gamma_beta = 1 / math.gamma(beta)
        fhistory = []
        yhistory = []

        # Get device from y0 (handle both tensor and tuple cases)
        if _is_tuple(y0):
            device = y0[0].device
        else:
            device = y0.device

        yn = _clone(y0)

        for k in range(N - 1):
            tn = tspan[k]
            f_k = func(tn, yn)
            fhistory.append(f_k)
            yhistory.append(yn)

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
            yn = _add(y0, weight_term)

        yhistory.append(yn)

        # release memory
        del fhistory
        del b_j_k_1
        return yn, yhistory


# TODO: need to revise it to accept tuple of tensors as input
def ForwardPredictor_wo_History(func, y0, beta, tspan, **options):
    with torch.no_grad():
        N = len(tspan)
        # print("N: ", N)
        h = (tspan[-1] - tspan[0]) / (N - 1)
        # print("h: ", h)
        gamma_beta = 1 / math.gamma(beta)
        fhistory = []
        device = y0.device
        yn = y0.clone()

        for k in range(N):
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


            # temp_product = torch.stack([b_j_k_1[i] * fhistory[i] for i in range(memory_k, k + 1)])
            # b_all_k = torch.sum(temp_product, dim=0)

            sample_product = b_j_k_1[memory_k] * fhistory[memory_k]
            b_all_k = torch.zeros_like(sample_product)  # Initialize accumulator with zeros of the same shape as the product
            # Loop through the range and accumulate results
            for i in range(memory_k, k + 1):
                b_all_k += b_j_k_1[i] * fhistory[i]

            yn = y0 + gamma_beta * b_all_k
    # release memory
    del fhistory
    del b_j_k_1, b_all_k
    del sample_product
    return yn, None

def BackwardMixedOrderGLmethod_f(func, y_aug, beta, tspan, yhistory_ori):
    with torch.no_grad():
        N = len(tspan)
        h = (tspan[N - 1] - tspan[0]) / (N - 1)
        h = torch.abs(h)
        _, adj_y0, adj_params0 = y_aug ### we will use yhistory_ori rather than compute y again

        if _is_tuple(adj_y0):
            device = adj_y0[0].device
        else:
            device = adj_y0.device

        adj_y = _clone(adj_y0)
        adj_params = _clone(adj_params0)

        c = torch.zeros(N + 1, dtype=torch.float64, device=device)
        c[0] = 1
        for j in range(1, N + 1):
            c[j] = (1 - (1 + beta) / j) * c[j - 1]
        h_power = torch.pow(h, beta)

        adjy_history = [adj_y]

        for k in range(1, N):
            tn = tspan[k]

            # Initialize right term with correct structure
            if _is_tuple(adj_y0):
                right = tuple(torch.zeros_like(comp) for comp in adj_y0)
            else:
                right = 0

            for j in range(1, k + 1):
                right = _add(right, _multiply(c[j], adjy_history[k - j]))

            y_ori = yhistory_ori[k]
            func_eval, f_term, jp_params = func(tn, (y_ori, adj_y, adj_params))

            f_h_term = _multiply(h_power, f_term)
            adj_y = _minus(f_h_term, right)

            adjy_history.append(adj_y)

            adj_params = adj_params + h * jp_params

    del adjy_history, yhistory_ori
    return adj_y, adj_params

def ForwardGLmethod_w_History(func,y0,beta,tspan,**options):
    with torch.no_grad():
        N = len(tspan)
        h = (tspan[N - 1] - tspan[0]) / (N - 1)
        h = torch.abs(h)
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

        return yn, y_history



def BackwardMixedOrderTrap_f(func, y_aug, beta, tspan, yhistory_ori):
    with torch.no_grad():
        N = len(tspan)
        h = (tspan[N - 1] - tspan[0]) / (N - 1)
        h = torch.abs(h)

        _, adj_y0, adj_params0 = y_aug  ### we will use yhistory_ori rather than compute y again

        if _is_tuple(adj_y0):
            device = adj_y0[0].device
        else:
            device = adj_y0.device

        adj_y = _clone(adj_y0)
        adj_params = _clone(adj_params0)

        c = torch.zeros(N + 1, dtype=torch.float64, device=device)
        c[0] = 1
        h_power = torch.pow(h, beta)
        gamma_factor = math.gamma(2 - beta)

        for j in range(1, N + 1):
            c[j] = (1 - (1 + beta) / j) * c[j - 1]

        adjy_history = [adj_y]

        for k in range(1, N):
            tn = tspan[k]

            # Initialize right term with correct structure
            if _is_tuple(adj_y):
                right = tuple(torch.zeros_like(comp) for comp in adj_y)
            else:
                right = 0

            for j in range(0, k):
                coeff = RLcoeffs(k, j, beta)
                # Handle tuple case
                right = _add(right, _multiply(coeff, adjy_history[j]))


            y_ori = yhistory_ori[k]
            func_eval, f_term, jp_params = func(tn, (y_ori, adj_y, adj_params))

            f_h_term = _multiply(h_power * gamma_factor, f_term)
            adj_y = _minus(f_h_term, right)
            adjy_history.append(adj_y)

            adj_params = adj_params + h * jp_params

    return adj_y, adj_params




def ForwardProduct_Trap_w_History(func,y0,beta,tspan,**options):
    with torch.no_grad():
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

            for j in range(0, k):
                coeff = RLcoeffs(k, j, beta)

                # Handle tuple case
                right = _add(right, _multiply(coeff, y_history[j]))

            # Calculate gamma * f(tn, yn) * h^beta term
            f_term = func(tn, yn)
            gamma_factor = math.gamma(2 - beta)
            h_power = torch.pow(h, beta)

            f_h_term = _multiply(h_power * gamma_factor, f_term)
            # Subtract right from f_h_term
            yn = _minus(f_h_term, right)
            y_history.append(yn)

    return yn, y_history


def BackwardMixedOrde_o(func, y_aug, beta, tspan, yhistory_ori):
    with torch.no_grad():
        N = len(tspan)
        h = (tspan[N - 1] - tspan[0]) / (N - 1)
        h = torch.abs(h)
        y0, adj_y0, adj_params0 = y_aug  ### we will use yhistory_ori rather than compute y again

        if _is_tuple(adj_y0):
            device = adj_y0[0].device
        else:
            device = adj_y0.device

        gamma_beta = 1 / math.gamma(beta)

        if True:#yhistory_ori is None:
            fy_history = []

        adj_y = _clone(adj_y0)
        adj_params = _clone(adj_params0)
        y = _clone(y0)

        for k in range(N-1):
            tn = tspan[k]


            func_eval, vjp_y, vjp_params = func(tn, (y, adj_y, adj_params))

            if yhistory_ori is not None and k<N:
                y = yhistory_ori[k+1]
            else:


                fy_history.append(func_eval)
                j_vals = torch.arange(0, k + 1, dtype=torch.float32, device=device).unsqueeze(1)
                b_j_k_1 = (fractional_pow(h, beta) / beta) * (
                        fractional_pow(k + 1 - j_vals, beta) - fractional_pow(k - j_vals, beta))

                # Initialize accumulator with correct structure (tensor or tuple)
                if _is_tuple(fy_history[0]):
                    b_all_k = tuple(torch.zeros_like(f_i) for f_i in fy_history[0])
                else:
                    b_all_k = torch.zeros_like(fy_history[0])

                # Loop through the range and accumulate results
                for i in range(0, k + 1):
                    b_all_k = _add(b_all_k, _multiply(b_j_k_1[i], fy_history[i]))

                # Final update step
                weight_term = _multiply(gamma_beta, b_all_k)
                y = _add(y0, weight_term)


            adj_y = _add(adj_y, _multiply(h, vjp_y))
            adj_params = adj_params + h * vjp_params

    del yhistory_ori, fy_history
    return adj_y, adj_params



SOLVERS = {"predictor-f":ForwardPredictor_w_History,
           "predictor-o":ForwardPredictor_w_History,
           "gl-f":ForwardGLmethod_w_History,
           "gl-o":ForwardGLmethod_w_History,
           "trap-f":ForwardProduct_Trap_w_History,
           "trap-o":ForwardProduct_Trap_w_History,
}