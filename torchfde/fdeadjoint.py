import torch
import math
import torch.nn as nn
from .utils_fde import _flatten, _flatten_convert_none_to_zeros,_check_inputs, _flat_to_shape
from .utils_fde import _addmul_inplace, _mul_inplace, _minusmul_inplace
from .utils_fde import _is_tuple, _clone, _add, _multiply, _minus, ReversedListView
from . import config


def fdeint_adjoint(func,y0,beta,t,step_size,method,options=None):

    # We need this in order to access the variables inside this module,
    # since we have no other way of getting variables along the execution path.
    if not isinstance(func, nn.Module):
        raise ValueError('func is required to be an instance of nn.Module.')

    tensor_input = False
    # Wrap single tensor inputs in a tuple for unified processing
    if torch.is_tensor(y0):
        class TupleFunc(nn.Module):

            def __init__(self, base_func):
                super(TupleFunc, self).__init__()
                self.base_func = base_func

            def forward(self, t, y):
                return (self.base_func(t, y[0]),)

        tensor_input = True
        y0 = (y0, ) # Convert tensor to tuple
        func = TupleFunc(func) # Wrap function to handle tensor input/output

    # Validate inputs and prepare for solving
    shapes, _, func, y0, tspan, method, beta = _check_inputs(func, y0, t, step_size, method, beta, SOLVERS_Forward)

    if options is None:
        options = {}

    # # Solve using adjoint method
    # flat_params = _flatten(func.parameters())
    # solution = FDEAdjointMethod.apply(*y0, func, beta, tspan, flat_params, method, options)

    # Get parameters
    params = find_parameters(func)
    n_state = len(y0)
    n_params = len(params)
    # Call FDEAdjointMethod with parameters
    solution = FDEAdjointMethod.apply(func, n_state, n_params, *y0, beta, tspan, method, *params, options)

    # Post-process solution based on tensor mode
    if config.TENSOR_MODE == 'concat':
        # CONCAT MODE: Always reshape the flattened solution back to original structure
        # Note: In adjoint method, inputs are always flattened/concatenated regardless of original type
        assert shapes is not None, 'for tuple, we need to provide shapes'
        solution = solution[0] # Extract from solver output
        solution = _flat_to_shape(solution, (), shapes) # Reshape to original structure
        if tensor_input:
            solution = solution[0] # If original input was a tensor, extract it from the tuple
    else:
        # NON-CONCAT MODE: Only unwrap if original input was a tensor
        if tensor_input:
            solution = solution[0]

    # Validate output type matches original input type
    if tensor_input:
        assert torch.is_tensor(solution)
    else:
        assert isinstance(solution, tuple)

    return solution

class FDEAdjointMethod(torch.autograd.Function):

    @staticmethod
    def forward(ctx, func, n_state, n_params, *args):
        # n_state, n_params are Python ints (not Tensors); their gradients should return None
        n_state = int(n_state)
        n_params = int(n_params)

        # Parse positional arguments: y0_1,...,y0_n, beta, tspan, method, p1,...,pm, options
        y0_tuple = tuple(args[:n_state])                               # Tensors
        beta = args[n_state]                                           # Tensor or float
        tspan = args[n_state + 1]                                      # Tensor
        method = args[n_state + 2]                                     # str/enum (not Tensor)
        func_params = tuple(args[n_state + 3 : n_state + 3 + n_params])  # Tensors (Parameters)
        options = args[n_state + 3 + n_params]                         # dict

        with torch.no_grad():
            ans, yhistory = SOLVERS_Forward[method](func=func, y0=y0_tuple, beta=beta, tspan=tspan, **options)

        # Check if gradients needed
        y0_needs_grad = any(t.requires_grad for t in y0_tuple)
        params_need_grad = any(p.requires_grad for p in func_params) if func_params else False

        ctx.n_state = n_state
        ctx.n_params = n_params
        if y0_needs_grad or params_need_grad:
            ctx.save_for_backward(tspan)
            ctx.ans = ans
            ctx.yhistory = yhistory
            ctx.func = func
            ctx.beta = beta
            ctx.method = method
            ctx.func_params = func_params
        else:
            del yhistory

        return ans

    @staticmethod
    def backward(ctx, *grad_output):

        """
        Backward pass - 计算梯度
        """
        # 早退：不需要反传时，返回正确数量的 None
        if not hasattr(ctx, 'yhistory'):
            n_state = ctx.n_state
            n_params = ctx.n_params
            grads = []
            grads.append(None)  # ode_func
            grads.append(None)  # n_state
            grads.append(None)  # n_params
            grads.extend([None] * n_state)  # y0_1,...,y0_n
            grads.append(None)  # beta
            grads.append(None)  # t_grid
            grads.append(None)  # method
            grads.extend([None] * n_params)  # p1,...,pm
            grads.append(None)  # memory
            return tuple(grads)


        tspan = ctx.saved_tensors[0]
        ans = ctx.ans
        yhistory = ctx.yhistory
        ans = tuple(ans)
        func = ctx.func
        beta = ctx.beta
        method = ctx.method
        func_params = ctx.func_params
        n_tensors = ctx.n_state

        # Create AugDynamics class similar to first file
        class AugDynamics:
            def __init__(self, func, n_tensors, func_params):
                self.func = func
                self.n_tensors = n_tensors
                self.f_params = func_params

            def __call__(self, t, y_aug):
                y, adj_y, adj_params = y_aug

                with torch.set_grad_enabled(True):
                    # detach and set requires_grad
                    y = tuple(y_.detach().requires_grad_(True) for y_ in y)
                    func_eval = self.func(t, y)

                    # Compute VJP
                    vjp_y_and_params = torch.autograd.grad(
                        func_eval,
                        y + self.f_params,
                        tuple(adj_y),
                        allow_unused=True,
                        retain_graph=False,
                        create_graph=False
                    )

                vjp_y = vjp_y_and_params[:self.n_tensors]
                vjp_params = vjp_y_and_params[self.n_tensors:]

                # Handle None gradients
                vjp_y = tuple(
                    torch.zeros_like(y_) if vjp_y_ is None else vjp_y_
                    for vjp_y_, y_ in zip(vjp_y, y)
                )

                vjp_params = tuple(
                    torch.zeros_like(p) if vp is None else vp
                    for vp, p in zip(vjp_params, self.f_params)
                )

                return (func_eval, vjp_y, vjp_params)

        augmented_dynamics = AugDynamics(func, n_tensors, func_params)
        tspan_flip = tspan.flip(0)

        if yhistory is not None:
            yhistory_flip = ReversedListView(yhistory)
        else:
            yhistory_flip = None

        with torch.no_grad():
            adj_y = grad_output

            # 初始化参数梯度
            if func_params:
                adj_params = tuple(torch.zeros_like(p) for p in func_params)
            else:
                adj_params = ()

            aug_y0 = (ans, adj_y, adj_params)

            adj_y, adj_params = SOLVERS_Backward[method](
                augmented_dynamics, aug_y0, beta,
                tspan_flip, yhistory_flip)

        # 在最后，确保清理所有局部变量
        del augmented_dynamics
        del yhistory_flip
        del yhistory  # 也要删除局部变量
        del ans
        del func
        del func_params
        del ctx.ans
        del ctx.yhistory
        del ctx.func
        del ctx.func_params
        del ctx.beta
        del ctx.method

        # Return gradients for each input: func, n_state, n_params, *y0, beta, tspan, method, *params, options
        return None, None, None, *adj_y, None, None, None, *adj_params, None
        #(func, n_state, n_params, *y0, beta, tspan, method, *params, options)


def backward_predictor(func, y_aug, beta, tspan, yhistory, **options):
    # mixed order predictor with beta and 1.
    with torch.no_grad():
        N = len(tspan)
        h = (tspan[-1] - tspan[0]) / (N - 1)
        h = torch.abs(h)
        gamma_beta = 1 / math.gamma(beta)
        # CHANGED: Pre-compute h^beta/beta for efficiency
        h_beta_over_beta = torch.pow(h, beta) / beta

        fadj_history = []
        if yhistory is None:  # CHANGED: Fixed condition (was "if True:")
            fy_history = []

        y0, adj_y0, adj_params0 = y_aug  ### we will use yhistory rather than compute y again
        if _is_tuple(y0):
            device = y0[0].device
        else:
            device = y0.device

        adj_y = _clone(adj_y0)
        adj_params = _clone(adj_params0)
        y = _clone(y0)

        for k in range(N - 1):
            tn = tspan[k]

            # CHANGED: Fixed memory handling to match corrected forward_predictor
            if 'memory' not in options or options['memory'] == -1:
                memory_length = k + 1  # Use all available history
            else:
                memory_length = min(options['memory'], k + 1)
                assert memory_length > 0, "memory must be greater than 0"

            # CHANGED: Corrected start_idx calculation
            start_idx = 0#max(0, k + 1 - memory_length)

            # CHANGED: j_vals now starts from start_idx instead of 0
            j_vals = torch.arange(start_idx, k + 1, dtype=torch.float32, device=device).unsqueeze(1)

            # CHANGED: Use torch.pow and pre-computed h_beta_over_beta
            b_j_k_1 = h_beta_over_beta * (
                    torch.pow(k + 1 - j_vals, beta) - torch.pow(k - j_vals, beta))

            func_eval, vjp_y, vjp_params = func(tn, (y, adj_y, adj_params))
            fadj_history.append(vjp_y)
            if yhistory is None:
                fy_history.append(func_eval)

            # CHANGED: Initialize accumulator properly
            convolution_sum = None

            # CHANGED: Loop through the correct range with proper indexing
            for j in range(start_idx, k + 1):
                local_idx = j - start_idx  # CHANGED: Use local index for b_j_k_1
                if convolution_sum is None:
                    convolution_sum = _multiply(b_j_k_1[local_idx], fadj_history[j])
                else:
                    # CHANGED: Use in-place operation for efficiency
                    convolution_sum = _addmul_inplace(convolution_sum, fadj_history[j], b_j_k_1[local_idx])

            # Final update step
            # CHANGED: Use in-place multiplication
            weight_term = _mul_inplace(convolution_sum, gamma_beta)
            adj_y = _add(adj_y0, weight_term)

            # Handle y update
            if yhistory is not None and k < N - 1:
                y = yhistory[k + 1]
            elif yhistory is None:
                # CHANGED: Reuse same pattern for y computation
                y_convolution_sum = None

                # CHANGED: Loop through the correct range with proper indexing
                for j in range(start_idx, k + 1):
                    local_idx = j - start_idx  # CHANGED: Use local index for b_j_k_1
                    if y_convolution_sum is None:
                        y_convolution_sum = _multiply(b_j_k_1[local_idx], fy_history[j])
                    else:
                        # CHANGED: Use in-place operation for efficiency
                        y_convolution_sum = _addmul_inplace(y_convolution_sum, fy_history[j], b_j_k_1[local_idx])

                # CHANGED: Use in-place multiplication
                y_weight_term = _mul_inplace(y_convolution_sum, gamma_beta)
                y = _add(y0, y_weight_term)

            # Update parameter gradients - already using in-place operation, good!
            if adj_params and vjp_params:
                for ap, vp in zip(adj_params, vjp_params):
                    ap.add_(vp, alpha=h)

        # release memory
        del fadj_history
        if yhistory is None:  # CHANGED: Only delete fy_history if it was created
            del fy_history
        del b_j_k_1
        return adj_y, adj_params


def forward_predictor(func, y0, beta, tspan, **options):
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
        h_beta_over_beta = torch.pow(h, beta) / beta

        fhistory = []
        yhistory = []
        # Get device from y0 (handle both tensor and tuple cases)
        if _is_tuple(y0):
            device = y0[0].device
        else:
            device = y0.device
        yn = _clone(y0)
        # yn = y0

        for k in range(N - 1):
            tn = tspan[k]
            f_k = func(tn, yn)
            fhistory.append(f_k)
            yhistory.append(yn)

            if 'memory' not in options or options['memory'] == -1:
                memory_length = k + 1  # Use all available history
            else:
                memory_length = min(options['memory'], k + 1)
                assert memory_length > 0, "memory must be greater than 0"

            start_idx = max(0, k + 1 - memory_length)

            j_vals = torch.arange(start_idx, k + 1, dtype=torch.float32, device=device).unsqueeze(1)

            b_j_k_1 = h_beta_over_beta * (
                    torch.pow(k + 1 - j_vals, beta) - torch.pow(k - j_vals, beta))

            convolution_sum = None

            for j in range(start_idx, k + 1):
                local_idx = j - start_idx  # CHANGED: Use local index for b_j_k_1
                if convolution_sum is None:
                    convolution_sum = _multiply(b_j_k_1[local_idx], fhistory[j])
                else:
                    # convolution_sum = _add(convolution_sum, _multiply(b_j_k_1[local_idx], fhistory[j]))
                    #_addmul_inplace(target, source, alpha):
                    # In-place fused multiply-add operation: target += alpha * source
                    convolution_sum = _addmul_inplace(convolution_sum, fhistory[j], b_j_k_1[local_idx])

            # Final update step
            # weight_term = _multiply(gamma_beta, convolution_sum)
            weight_term = _mul_inplace(convolution_sum, gamma_beta)
            yn = _add(y0, weight_term)

        yhistory.append(yn)
        # release memory
        del fhistory
        del b_j_k_1
        return yn, yhistory


def backward_gl(func, y_aug, beta, tspan, yhistory, **options):
    with torch.no_grad():
        N = len(tspan)
        h = (tspan[N - 1] - tspan[0]) / (N - 1)
        h = torch.abs(h)
        _, adj_y0, adj_params0 = y_aug  ### we will use yhistory rather than compute y again
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

        # CHANGED: Use adj_y_current for clarity
        adj_y_current = _clone(adj_y0)
        adjy_history = [adj_y_current]

        # CHANGED: Fixed loop range from range(1, N) to range(N - 1)
        for k in range(N - 1):
            # CHANGED: Use tspan[k] for current time
            t_k = tspan[k]

            # CHANGED: Get the corresponding y from history at current time
            y_current = yhistory[k]

            # CHANGED: Evaluate function at current time with current states
            func_eval, vjp_y, vjp_params = func(t_k, (y_current, adj_y_current, adj_params))

            # CHANGED: Add memory handling
            if 'memory' not in options or options['memory'] == -1:
                memory_length = k + 1  # Use all available history
            else:
                memory_length = min(options['memory'], k + 1)
                assert memory_length > 0, "memory must be greater than 0"

            start_idx = 0#max(0, k + 1 - memory_length)

            # CHANGED: Initialize convolution_sum properly
            convolution_sum = None

            # CHANGED: Fix summation indices and coefficients
            # The sum should be Σ c_{k+1-j} * adjy_j for j from start_idx to k
            for j in range(start_idx, k + 1):
                # CHANGED: Use correct coefficient index (k+1-j instead of j)
                coefficient_idx = k + 1 - j

                if convolution_sum is None:
                    convolution_sum = _multiply(c[coefficient_idx], adjy_history[j])
                else:
                    # CHANGED: Use in-place operation for efficiency
                    convolution_sum = _addmul_inplace(convolution_sum, adjy_history[j], c[coefficient_idx])

            # Compute adj_y_{k+1} = h^α * vjp_y - convolution_sum
            # f_h_term = _multiply(h_power, vjp_y)
            # adj_y_current = _minus(f_h_term, convolution_sum)
            adj_y_current = _minusmul_inplace(convolution_sum, vjp_y, h_power)

            # Store adj_y_{k+1} in history
            adjy_history.append(adj_y_current)

            # Update parameter gradients - already using in-place operation, good!
            if adj_params and vjp_params:
                for ap, vp in zip(adj_params, vjp_params):
                    ap.add_(vp, alpha=h)

        # CHANGED: Return adj_y_current instead of adj_y
        del adjy_history, yhistory
        return adj_y_current, adj_params



def forward_gl(func, y0, beta, tspan, **options):
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

        # CHANGED: Compute h^beta once outside the loop for efficiency
        h_power = torch.pow(h, beta)

        # CHANGED: Use y_current for clarity and consistency
        y_current = _clone(y0)
        y_history = [y_current]

        # CHANGED: Loop range from range(1, N) to range(N - 1) to match correct algorithm
        for k in range(N - 1):
            # CHANGED: Use tspan[k] for current time (not tspan[k] when k starts from 1)
            t_k = tspan[k]

            # CHANGED: Evaluate function at current time with current y (not future y)
            f_k = func(t_k, y_current)

            # CHANGED: Add memory handling
            if 'memory' not in options or options['memory'] == -1:
                memory_length = k + 1  # Use all available history
            else:
                memory_length = min(options['memory'], k + 1)
                assert memory_length > 0, "memory must be greater than 0"

            start_idx = max(0, k + 1 - memory_length)

            # CHANGED: Initialize convolution_sum properly
            convolution_sum = None

            # CHANGED: Fix summation indices and coefficients
            # The sum should be Σ c_{k+1-j} * y_j for j from start_idx to k
            for j in range(start_idx, k + 1):
                # CHANGED: Use correct coefficient index (k+1-j instead of j)
                coefficient_idx = k + 1 - j

                if convolution_sum is None:
                    convolution_sum = _multiply(c[coefficient_idx], y_history[j])
                else:
                    # CHANGED: Use in-place operation for efficiency
                    convolution_sum = _addmul_inplace(convolution_sum, y_history[j], c[coefficient_idx])

            # # CHANGED: Move h_power multiplication outside loop and use it here
            # f_h_term = _multiply(h_power, f_k)
            # # Compute y_{k+1} = h^α * f(t_k, y_k) - convolution_sum
            # y_current = _minus(f_h_term, convolution_sum)

            #In-place fused multiply-add operation: target = -target + alpha * source
            y_current = _minusmul_inplace(convolution_sum, f_k, h_power)

            # Store y_{k+1} in history
            y_history.append(y_current)

        return y_current, y_history  # CHANGED: Fixed - return y_current instead of yn


def backward_trap(func, y_aug, beta, tspan, yhistory_ori, **options):
    with torch.no_grad():
        N = len(tspan)
        h = (tspan[N - 1] - tspan[0]) / (N - 1)
        h = torch.abs(h)

        # CHANGED: Pre-compute h^beta * Gamma(2-beta) for efficiency
        h_alpha_gamma = torch.pow(h, beta) * math.gamma(2 - beta)
        one_minus_beta = 1 - beta

        _, adj_y0, adj_params0 = y_aug  ### we will use yhistory_ori rather than compute y again
        if _is_tuple(adj_y0):
            device = adj_y0[0].device
        else:
            device = adj_y0.device

        adj_params = _clone(adj_params0)

        # CHANGED: Removed unused c array computation
        # CHANGED: Use adj_y_current for clarity
        adj_y_current = _clone(adj_y0)
        adjy_history = [adj_y_current]

        # CHANGED: Fixed loop range from range(1, N) to range(N - 1)
        for k in range(N - 1):
            # CHANGED: Use tspan[k] for current time
            t_k = tspan[k]

            # CHANGED: Get the corresponding y from history at current time
            y_current = yhistory_ori[k]

            # CHANGED: Evaluate function at current time with current states
            func_eval, vjp_y, vjp_params = func(t_k, (y_current, adj_y_current, adj_params))

            # CHANGED: Add memory handling
            if 'memory' not in options or options['memory'] == -1:
                memory_length = k + 1  # Use all available history
            else:
                memory_length = min(options['memory'], k + 1)
                assert memory_length > 0, "memory must be greater than 0"

            start_idx = 0#max(0, k + 1 - memory_length)

            # CHANGED: Compute A_{j,k+1} weights correctly instead of RLcoeffs
            j_vals = torch.arange(start_idx, k + 1, dtype=torch.float32, device=device)

            # Compute A_{j,k+1} weights (same as forward_trap)
            kjp2 = torch.pow(k + 2 - j_vals, one_minus_beta)
            kj = torch.pow(k - j_vals, one_minus_beta)
            kjp1 = torch.pow(k + 1 - j_vals, one_minus_beta)

            # General formula for j >= 1
            A_j_kp1 = kjp2 + kj - 2 * kjp1

            # CHANGED: Special handling for j=0 if it's in the range
            if start_idx == 0:
                k_power = torch.pow(torch.tensor(k, dtype=torch.float32, device=device), one_minus_beta)
                kp1_neg_alpha = torch.pow(torch.tensor(k + 1, dtype=torch.float32, device=device), -beta)
                A_j_kp1[0] = k_power - (k + beta) * kp1_neg_alpha

            # CHANGED: Initialize convolution_sum properly
            convolution_sum = None

            # CHANGED: Accumulate with correct indexing
            for j in range(start_idx, k + 1):
                local_idx = j - start_idx  # Index into A_j_kp1 array

                if convolution_sum is None:
                    convolution_sum = _multiply(A_j_kp1[local_idx], adjy_history[j])
                else:
                    # CHANGED: Use in-place operation for efficiency
                    convolution_sum = _addmul_inplace(convolution_sum, adjy_history[j], A_j_kp1[local_idx])

            # Compute adj_y_{k+1} = Γ(2-α) * h^α * vjp_y - convolution_sum
            # f_h_term = _multiply(h_alpha_gamma, vjp_y)
            # adj_y_current = _minus(f_h_term, convolution_sum)

            adj_y_current = _minusmul_inplace(convolution_sum, vjp_y, h_alpha_gamma)


            # Store adj_y_{k+1} in history
            adjy_history.append(adj_y_current)

            # Update parameter gradients - already using in-place operation, good!
            if adj_params and vjp_params:
                for ap, vp in zip(adj_params, vjp_params):
                    ap.add_(vp, alpha=h)

        # CHANGED: Add memory cleanup
        del adjy_history, yhistory_ori
        # CHANGED: Return adj_y_current instead of adj_y
        return adj_y_current, adj_params

def forward_trap(func, y0, beta, tspan, **options):
    with torch.no_grad():
        N = len(tspan)
        h = (tspan[N - 1] - tspan[0]) / (N - 1)

        # CHANGED: Pre-compute h^beta * Gamma(2-beta) for efficiency
        h_alpha_gamma = torch.pow(h, beta) * math.gamma(2 - beta)
        one_minus_beta = 1 - beta

        # Get device from y0 (handle both tensor and tuple cases)
        if _is_tuple(y0):
            device = y0[0].device
        else:
            device = y0.device

        # CHANGED: Removed unused c array computation
        # CHANGED: Use y_current for clarity
        y_current = _clone(y0)
        y_history = [y0]  # CHANGED: Store y0 not yn as first element

        # CHANGED: Fixed loop range from range(1, N) to range(N - 1)
        for k in range(N - 1):
            # CHANGED: Use tspan[k] for current time
            t_k = tspan[k]

            # CHANGED: Evaluate function at current time with current y
            f_k = func(t_k, y_current)

            # CHANGED: Add memory handling
            if 'memory' not in options or options['memory'] == -1:
                memory_length = k + 1  # Use all available history
            else:
                memory_length = min(options['memory'], k + 1)
                assert memory_length > 0, "memory must be greater than 0"

            start_idx = max(0, k + 1 - memory_length)

            # CHANGED: Compute A_{j,k+1} weights correctly instead of RLcoeffs
            j_vals = torch.arange(start_idx, k + 1, dtype=torch.float32, device=device)

            # Compute A_{j,k+1} weights
            kjp2 = torch.pow(k + 2 - j_vals, one_minus_beta)
            kj = torch.pow(k - j_vals, one_minus_beta)
            kjp1 = torch.pow(k + 1 - j_vals, one_minus_beta)

            # General formula for j >= 1
            A_j_kp1 = kjp2 + kj - 2 * kjp1

            # CHANGED: Special handling for j=0 if it's in the range
            if start_idx == 0:
                k_power = torch.pow(torch.tensor(k, dtype=torch.float32, device=device), one_minus_beta)
                kp1_neg_alpha = torch.pow(torch.tensor(k + 1, dtype=torch.float32, device=device), -beta)
                A_j_kp1[0] = k_power - (k + beta) * kp1_neg_alpha

            # CHANGED: Initialize convolution_sum properly
            convolution_sum = None

            # CHANGED: Accumulate with correct indexing
            for j in range(start_idx, k + 1):
                local_idx = j - start_idx  # Index into A_j_kp1 array

                if convolution_sum is None:
                    convolution_sum = _multiply(A_j_kp1[local_idx], y_history[j])
                else:
                    # CHANGED: Use in-place operation for efficiency
                    convolution_sum = _addmul_inplace(convolution_sum, y_history[j], A_j_kp1[local_idx])

            # # CHANGED: Compute y_{k+1} correctly
            # f_term = _multiply(h_alpha_gamma, f_k)
            # # CHANGED: Use _minus or multiply by -1 properly
            # y_current = _minus(f_term, convolution_sum)

            # In-place fused multiply-add operation: target = -target + alpha * source
            y_current = _minusmul_inplace(convolution_sum, f_k, h_alpha_gamma)


            # Store y_{k+1} in history
            y_history.append(y_current)

        return y_current, y_history  # CHANGED: Return y_current instead of yn

def backward_euler_w_history(func, y_aug, beta, tspan, yhistory):
    with torch.no_grad():
        N = len(tspan)
        # print('N = len(tspan)', N, tspan)
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

        # return tuple(y_i.clone() for y_i in adj_y0), tuple(y_i.clone() for y_i in adj_params0)

        for k in range(N-1):
            tn = tspan[k]


            func_eval, vjp_y, vjp_params = func(tn, (y, adj_y, adj_params))
            y = yhistory[k + 1]

            ## We assume having the full yhistory
            ## We do not consider the following case any more.
            # if yhistory is not None and k<N:
            #     y = yhistory[k+1]
            # else:
            #     fy_history.append(func_eval)
            #     j_vals = torch.arange(0, k + 1, dtype=torch.float32, device=device).unsqueeze(1)
            #     b_j_k_1 = (torch.pow(h, beta) / beta) * (
            #             torch.pow(k + 1 - j_vals, beta) - torch.pow(k - j_vals, beta))
            #
            #     # Initialize accumulator with correct structure (tensor or tuple)
            #     if _is_tuple(fy_history[0]):
            #         b_all_k = tuple(torch.zeros_like(f_i) for f_i in fy_history[0])
            #     else:
            #         b_all_k = torch.zeros_like(fy_history[0])
            #
            #     # Loop through the range and accumulate results
            #     for i in range(0, k + 1):
            #         b_all_k = _add(b_all_k, _multiply(b_j_k_1[i], fy_history[i]))
            #
            #     # Final update step
            #     weight_term = _multiply(gamma_beta, b_all_k)
            #     y = _add(y0, weight_term)

            # adj_y = _add(adj_y, _multiply(h, vjp_y))
            adj_y = _addmul_inplace(adj_y, vjp_y, h)


            # Update parameter gradients using tuple comprehension
            # 更新参数梯度
            if adj_params and vjp_params:
                for ap, vp in zip(adj_params, vjp_params):
                    ap.add_(vp, alpha=h)  # 直接修改 tuple 中的张量

    del yhistory, fy_history
    return adj_y, adj_params

def find_parameters(module):

    assert isinstance(module, nn.Module)

    # If called within DataParallel, parameters won't appear in module.parameters().
    if getattr(module, '_is_replica', False):

        def find_tensor_attributes(module):
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v) and v.requires_grad]
            return tuples

        gen = module._named_members(get_members_fn=find_tensor_attributes)
        return [param for _, param in gen]
    else:
        return list(module.parameters())



# forward_gl_compiled = torch.compile(forward_gl)
# backward_gl_compiled = torch.compile(backward_gl)
# forward_predictor_compiled = torch.compile(forward_predictor)
# backward_predictor_compiled = torch.compile(backward_predictor)
# forward_trap_compiled = torch.compile(forward_trap)
# backward_trap_compiled = torch.compile(backward_trap)


forward_gl_compiled = forward_gl#torch.compile(forward_gl)
backward_gl_compiled = backward_gl#torch.compile(backward_gl)
forward_predictor_compiled = forward_predictor#torch.compile(forward_predictor)
backward_predictor_compiled = backward_predictor#torch.compile(backward_predictor)
forward_trap_compiled = forward_trap#torch.compile(forward_trap)
backward_trap_compiled = backward_trap#torch.compile(backward_trap)

backward_euler_w_history_compiled = backward_euler_w_history#torch.compile(backward_euler_w_history)



SOLVERS_Forward = {
            "predictor-f":forward_predictor_compiled,
           "predictor-o":forward_predictor_compiled,
           "gl-f":forward_gl_compiled,
           "gl-o":forward_gl_compiled,
           "trap-f":forward_trap_compiled,
           "trap-o":forward_trap_compiled,
            # "euler":forward_euler_w_history_compiled,
}

SOLVERS_Backward = {"predictor-f":backward_predictor_compiled,
           "predictor-o":backward_euler_w_history_compiled,
           "gl-f":backward_gl_compiled,
           "gl-o":backward_euler_w_history_compiled,
           "trap-f":backward_trap_compiled,
           "trap-o":backward_euler_w_history_compiled,
            # "euler": backward_euler_w_history_compiled,
}
