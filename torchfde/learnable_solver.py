import torch.nn.functional as F
import math
from typing import Callable, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from .utils_fde import _check_inputs_tensorinput
from .explicit_solver import fractional_pow


class LearnbleFDEINT(nn.Module):
    """
    Neural solver for integral equations using learnable attention mechanisms.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 64, dropout: float = 0.1, method: str = 'AttentionKernel_simple'):
        super(LearnbleFDEINT, self).__init__()
        self.state_dim = state_dim

        # Attention kernel for computing weights
        self.attention_kernel = None

        # Attention kernel for computing weights
        self.attention_kernel = None
        """
        Initialize the neural solver for later use.
        """
        if method == "AttentionKernel":
            self.attention_kernel = AttentionKernel(
                state_dim=state_dim,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
        elif method == "AttentionKernel_simple":
            self.attention_kernel = AttentionKernel_simple(
                state_dim=state_dim,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
        elif method == "AttentionKernel_Position":
            self.attention_kernel = AttentionKernel_Position(
                state_dim=state_dim,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
        else:
            raise ValueError("No learnable solver specified. Please specify a way to compute the kernel.")

    def forward(self,
                func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                y0: torch.Tensor,
                beta: torch.Tensor,
                t: float = 1.0,
                step_size: float = 0.1,
                method: str = 'AttentionKernel',
                **options):
        """
        Forward pass for solving the integral equation.

        Args:
            func: Function that computes f(t, y)
            y0: Initial condition
            beta: Parameter for fractional derivative
            tspan: Time points for integration

        Returns:
            Solution at all time points [batch_size, len(tspan), state_dim]
        """

        # Check inputs
        func, y0, tspan, method, beta = _check_inputs_tensorinput(func, y0, t, step_size, method, beta, SOLVERS)
        if options is None:
            options = {}
        # Ensure y0 is a tensor

        device = y0.device
        batch_size, state_dim = y0.shape

        # # Handle the learnable solver differently
        # if self.attention_kernel is None:
        #     self.initialize_attention_kernel(method, state_dim, device, hidden_dim=64, dropout=0.1)

        # Convert tspan to tensor if it's not already
        if not isinstance(tspan, torch.Tensor):
            tspan = torch.tensor(tspan, dtype=torch.float32, device=device)
        tspan = tspan.to(device)

        N = len(tspan)
        h = (tspan[-1] - tspan[0]) / (N - 1)

        # Initialize lists with the initial condition
        y_list = [y0]
        f_list = [func(tspan[0], y0)]

        # Integration loop
        for k in range(1, N):
            # Current time
            tk = tspan[k]
            prev_tk = tspan[k - 1]

            # Memory constraints if specified
            # in this version, we only consider full memory
            if False:#'memory' in options:
                memory_length = options['memory']
                memory_start = max(0, k - memory_length)
            else:
                memory_start = 0
            # memory_start = 0

            # Create mask for memory constraints
            curr_mask = torch.zeros(k, device=device)
            curr_mask[memory_start:k] = 1.0  # [k]

            #check if method is "AttentionKernel":
            if method == "AttentionKernel" or method == "AttentionKernel_simple" or method == "AttentionKernel_Position":
                # Extract relevant history
                t_history = tspan[:k]

                # Create history tensor by stacking list elements
                y_history_tensor = torch.stack(y_list, dim=-1)  # [batch_size, state_dim, k]
                f_history_tensor = torch.stack(f_list, dim=-1)  # [batch_size, state_dim, k]

                options['beta'] = beta
                options['h'] = h
                # Compute attention weights
                attn_weights = self.attention_kernel(
                    prev_tk,  # Current time
                    t_history,  # History of time points
                    y_list[-1],  # Current state (last in list)
                    y_history_tensor,  # History of states
                    **options
                )  # [batch_size, k]

                # Apply memory constraints to attention weights
                attn_weights = attn_weights * curr_mask  # [batch_size, k]

                # Renormalize weights
                # attn_weights = attn_weights / (attn_weights.sum(-1, keepdim=True) + 1e-10)  # [batch_size, k]

                # Compute weighted sum of function values
                weighted_sum = torch.sum(
                    attn_weights.unsqueeze(1) * f_history_tensor,  # [batch_size, state_dim, k]
                    dim=2  # Sum along the time dimension
                )  # [batch_size, state_dim]

            else:
                # alert user
                print("No attention kernel used. Please specify a way to compute the kernel.")

            # Compute new state
            y_new = y0 + weighted_sum * h

            # Compute new function value
            f_new = func(tk, y_new)

            # Append to lists
            y_list.append(y_new)
            f_list.append(f_new)
        # print(attn_weights)
        # Return final state
        return y_list[-1]  # [batch_size, state_dim]



class AttentionKernel_simple(nn.Module):
    """
    Learnable attention kernel for neural integral equations.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super(AttentionKernel_simple, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

    def forward(self,
                t_current: torch.Tensor,
                t_history: torch.Tensor,
                y_current: torch.Tensor,
                y_history: torch.Tensor,
                **options) -> torch.Tensor:
        """
        Forward pass for the attention kernel.

        Args:
            t_current: Current time points [1]
            t_history: History of time points [seq_len, 1]
            y_current: Current state [batch_size, state_dim]
            y_history: History of states [batch_size, state_dim， seq_len]
            mask: Optional mask tensor of shape [batch_size, 1, seq_len]

        Returns:
            Attention weights [batch_size, seq_len]
        """
        batch_size, state_dim, seq_len = y_history.shape
        device = y_current.device
        # Calculate time difference and features
        time_diff = (t_current - t_history).abs()  # [batch_size, 1, seq_len]
        time_features = 1  # torch.exp(time_diff)  # [batch_size, 1, seq_len]
        # change it to

        # Reshape current state for matrix multiplication
        # y_current: [batch_size, state_dim] -> [batch_size, state_dim, 1]
        y_current_expanded = y_current.unsqueeze(-1)  # [batch_size, state_dim, 1]

        # Calculate attention scores using matrix multiplication
        # [batch_size, 1, state_dim] @ [batch_size, state_dim, seq_len] = [batch_size, 1, seq_len]
        attn_scores = torch.matmul(
            y_current_expanded.transpose(1, 2),  # [batch_size, 1, state_dim]
            y_history  # [batch_size, state_dim, seq_len]
        ) / math.sqrt(state_dim)  # [batch_size, 1, seq_len]

        # Apply time features to attention scores
        attn_scores = attn_scores * time_features  # [batch_size, 1, seq_len]

        # Softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1).squeeze(1)  # [batch_size, seq_len]

        # add fractional decay term to attention weights
        if options is not None:
            if 'beta' in options and 'h' in options:
                beta = options['beta']
                h = options['h']
                j_vals = torch.arange(0, seq_len, dtype=torch.float32, device=device).unsqueeze(1)
                b_j_k_1 = (fractional_pow(h, beta) / beta) * (fractional_pow(seq_len - j_vals, beta))

                # multiply attention weights with b_j_k_1
                attn_weights = attn_weights + b_j_k_1.squeeze(-1)  # [batch_size, seq_len]

        return attn_weights


class AttentionKernel(nn.Module):
    """
    Learnable attention kernel for neural integral equations.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 64, dropout: float = 0.1, pos_encoding: str = 'sinusoidal', max_seq_len: int = 1000):
        super(AttentionKernel, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # Learnable projection matrices for query, key
        self.W_q = nn.Linear(state_dim, hidden_dim)
        self.W_k = nn.Linear(state_dim, hidden_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                t_current: torch.Tensor,
                t_history: torch.Tensor,
                y_current: torch.Tensor,
                y_history: torch.Tensor,
                **options) -> torch.Tensor:
        batch_size, state_dim, seq_len = y_history.shape
        device = y_current.device


        # print(self.W_q.weight)
        # Project current state to query
        q = self.W_q(y_current).unsqueeze(1)  # [batch_size, 1, hidden_dim]

        # Project history states to keys
        k = self.W_k(y_history.permute(0, 2, 1))  # [batch_size, seq_len, hidden_dim]

        # Incorporate time encoding into keys
        k = k #+ time_encoding

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.hidden_dim)  # [batch_size, num_heads, 1, seq_len]


        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1).squeeze(1)  # [batch_size, 1, seq_len]
        attn_weights = self.dropout(attn_weights)


        # add fractional decay term to attention weights
        if options is not None:
            if 'beta' in options and 'h' in options:
                beta = options['beta']
                h = options['h']
                j_vals = torch.arange(0, seq_len, dtype=torch.float32, device=device).unsqueeze(1)
                b_j_k_1 = (fractional_pow(h, beta) / beta) * (fractional_pow(seq_len - j_vals, beta))

                # multiply attention weights with b_j_k_1
                attn_weights = attn_weights + b_j_k_1.squeeze(-1)  # [batch_size, seq_len]

        return attn_weights

class AttentionKernel_Position(nn.Module):
    """
    Learnable attention kernel for neural integral equations.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 64, dropout: float = 0.1, pos_encoding: str = 'sinusoidal', max_seq_len: int = 100):
        super(AttentionKernel_Position, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # Learnable projection matrices for query, key
        self.W_q = nn.Linear(state_dim, hidden_dim)
        self.W_k = nn.Linear(state_dim, hidden_dim)

        # Positional encoding
        self.positional_encoding = PositionalEncoding.get_encoding(
            pos_encoding, state_dim, max_seq_len, dropout
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                t_current: torch.Tensor,
                t_history: torch.Tensor,
                y_current: torch.Tensor,
                y_history: torch.Tensor,
                **options) -> torch.Tensor:

        batch_size, state_dim, seq_len = y_history.shape
        device = y_current.device

        # Ensure t_current has the right shape [batch_size, 1]
        if t_current.dim() == 0:  # scalar
            t_current = t_current.unsqueeze(0).unsqueeze(0).expand(batch_size, 1)
        elif t_current.dim() == 1:  # [batch_size] or [1]
            if t_current.size(0) == 1:
                t_current = t_current.unsqueeze(0).expand(batch_size, 1)
            else:
                t_current = t_current.unsqueeze(1)  # [batch_size, 1]

        # Ensure t_history has the right shape [batch_size, seq_len]
        if t_history.dim() == 1:  # [seq_len]
            t_history = t_history.unsqueeze(0).expand(batch_size, -1)  # [batch_size, seq_len]

        # Transpose history states for positional encoding
        y_history_t = y_history.transpose(1, 2)  # [batch_size, seq_len, state_dim]

        # Apply positional encoding
        time_encoding = self.positional_encoding(y_history_t, t_history)  # [batch_size, seq_len, state_dim]

        # # Project current state to query
        # q = self.W_q(y_current).unsqueeze(1)  # [batch_size, 1, hidden_dim]

        # Project history states to keys
        k = self.W_k(y_history.permute(0, 2, 1))  # [batch_size, seq_len, hidden_dim]

        # Incorporate time encoding into keys
        k = k + time_encoding

        q = k[:, -1:,:]

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.hidden_dim)  # [batch_size, num_heads, 1, seq_len]


        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1).squeeze(1)  # [batch_size, 1, seq_len]
        attn_weights = self.dropout(attn_weights)


        # add fractional decay term to attention weights
        if options is not None:
            if 'beta' in options and 'h' in options:
                beta = options['beta']
                h = options['h']
                j_vals = torch.arange(0, seq_len, dtype=torch.float32, device=device).unsqueeze(1)
                b_j_k_1 = (fractional_pow(h, beta) / beta) * (fractional_pow(seq_len - j_vals, beta))

                # multiply attention weights with b_j_k_1
                attn_weights = attn_weights + b_j_k_1.squeeze(-1)  # [batch_size, seq_len]

        return attn_weights



class PositionalEncoding:
    """Factory class for different types of positional encodings"""

    @staticmethod
    def get_encoding(encoding_type, state_dim, max_seq_len=1000, dropout=0.1):
        if encoding_type == 'sinusoidal':
            return SinusoidalPositionalEncoding(state_dim, max_seq_len, dropout)
        elif encoding_type == 'relative':
            return RelativePositionalEncoding(state_dim, max_seq_len, dropout)
        elif encoding_type == 'learned':
            return LearnablePositionalEncoding(state_dim, max_seq_len, dropout)
        elif encoding_type == 'none':
            return NoPositionalEncoding()
        else:
            raise ValueError(f"Unknown positional encoding type: {encoding_type}")


class NoPositionalEncoding(nn.Module):
    """No positional encoding - simply passes the input through"""

    def __init__(self):
        super().__init__()

    def forward(self, x, times=None):
        return x


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from the paper 'Attention Is All You Need'
    """

    def __init__(self, d_model, max_len=1000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a table of positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, times=None):
        """
        Args:
            x: Tensor [batch_size, seq_len, state_dim] or [batch_size, state_dim, seq_len]
            times: Optional tensor of time points
        """
        # If x has shape [batch_size, state_dim, seq_len], transpose it
        if x.size(1) != x.size(2) and x.size(1) == self.pe.size(1):
            x = x.transpose(1, 2)
            transpose_back = True
        else:
            transpose_back = False

        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].to(x.device).unsqueeze(0)
        x = self.dropout(x)

        # Transpose back if needed
        if transpose_back:
            x = x.transpose(1, 2)

        return x


class RelativePositionalEncoding(nn.Module):
    """
    Relative positional encoding based on time differences
    """

    def __init__(self, d_model, max_len=1000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.encoding_fn = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, d_model)
        )

    def forward(self, x, times):
        """
        Args:
            x: Tensor [batch_size, seq_len, state_dim] or [batch_size, state_dim, seq_len]
            times: Tensor of time points [batch_size, seq_len] or [seq_len]
        """
        # Handle different input shapes
        if x.size(1) != x.size(2) and x.size(1) == x.size(-1):
            x = x.transpose(1, 2)
            transpose_back = True
        else:
            transpose_back = False

        # Handle different time shapes
        if times.dim() == 1:  # [seq_len]
            times = times.unsqueeze(0).expand(x.size(0), -1)  # [batch_size, seq_len]

        # Calculate time differences (relative positions)
        batch_size, seq_len = times.size()
        times_matrix = times.unsqueeze(-1).expand(batch_size, seq_len, 1)  # [batch_size, seq_len, 1]

        # Encode time differences
        time_encoding = self.encoding_fn(times_matrix)  # [batch_size, seq_len, d_model]

        # Add to input
        x = x + time_encoding
        x = self.dropout(x)

        # Transpose back if needed
        if transpose_back:
            x = x.transpose(1, 2)

        return x


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding where the position embeddings are learned parameters
    """

    def __init__(self, d_model, max_len=1000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x, times=None):
        """
        Args:
            x: Tensor [batch_size, seq_len, state_dim] or [batch_size, state_dim, seq_len]
            times: Optional tensor of time points (not used in this encoding)
        """
        # If x has shape [batch_size, state_dim, seq_len], transpose it
        if x.size(1) != x.size(2) and x.size(1) == self.pos_embedding.size(-1):
            x = x.transpose(1, 2)
            transpose_back = True
        else:
            transpose_back = False

        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]
        x = self.dropout(x)

        # Transpose back if needed
        if transpose_back:
            x = x.transpose(1, 2)

        return x


SOLVERS = {"AttentionKernel_simple":AttentionKernel_simple,
          "AttentionKernel":AttentionKernel,
         "AttentionKernel_Position":AttentionKernel_Position
}