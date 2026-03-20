import torch
import torch.nn as nn
import torch.nn.functional as F
import torchcvnn.nn.modules as c_nn


class PolyCtoR(nn.Module):
    def __init__(self, order=3):
        super().__init__()
        self.order = order
        # Precompute exponent indices for vectorized computation (exclude constant term)
        exps = []
        for n in range(1, order + 1):
            for i in range(n + 1):
                j = n - i
                exps.append((i, j))
        # store as LongTensors for indexing
        if len(exps) == 0:
            self.register_buffer("_exp_i", torch.empty(0, dtype=torch.long))
            self.register_buffer("_exp_j", torch.empty(0, dtype=torch.long))
        else:
            exp_i = torch.tensor([e[0] for e in exps], dtype=torch.long)
            exp_j = torch.tensor([e[1] for e in exps], dtype=torch.long)
            self.register_buffer("_exp_i", exp_i)
            self.register_buffer("_exp_j", exp_j)
        self.size = len(exps)
        self.poly = nn.Linear(self.size, 1, bias=True)

    def forward(self, x):
        size_cpy = x.size()
        x = torch.view_as_real(x)  # (..., 2)
        x = x.reshape(-1, 2)
        x_real = x[:, 0]  # (N,)
        x_imag = x[:, 1]  # (N,)

        # If order is zero (no terms), return zero output
        if self.size == 0:
            out = self.poly(torch.empty((x_real.size(0), 0), device=x_real.device, dtype=x_real.dtype))
            out = out.view(*size_cpy)
            return out

        # Compute powers up to max exponent once
        max_i = int(self._exp_i.max().item())
        max_j = int(self._exp_j.max().item())
        # shape (N, max_i+1)
        real_pows = torch.stack([x_real ** p for p in range(max_i + 1)], dim=1)
        # shape (N, max_j+1)
        imag_pows = torch.stack([x_imag ** p for p in range(max_j + 1)], dim=1)

        # Gather required powers and multiply: for k-th term use real_pows[:, exp_i[k]] * imag_pows[:, exp_j[k]]
        # exp indices are on CPU but small; move to device for gather
        exp_i = self._exp_i.to(real_pows.device)
        exp_j = self._exp_j.to(imag_pows.device)
        # real_terms: (N, size)
        real_terms = real_pows.gather(1, exp_i.unsqueeze(0).expand(real_pows.size(0), -1))
        imag_terms = imag_pows.gather(1, exp_j.unsqueeze(0).expand(imag_pows.size(0), -1))
        x_poly = real_terms * imag_terms
        out = self.poly(x_poly)
        out = out.view(*size_cpy)
        return out

class MLPCtoR(nn.Module):
    def __init__(self, input_size=2, hidden_sizes=None, output_size=1):
        super(MLPCtoR, self).__init__()

        # Default hidden sizes if none provided
        if hidden_sizes is None:
            hidden_sizes = [8, 16]

        # Build the network layers
        layers = []
        current_size = input_size

        # Add hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_size, hidden_size, bias=True))
            layers.append(nn.ReLU())
            current_size = hidden_size

        # Add output layer
        layers.append(nn.Linear(current_size, output_size, bias=True))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        size_cpy = x.size()  # B, C, H, W or B, C
        x = torch.view_as_real(x)  # B, C, H, W, 2 or B, C, 2
        if x.dim() not in [3, 5]:
            raise ValueError("Unexpected tensor shape")

        x = x.reshape(-1, 2)  # (B*X, 2)

        x = self.mlp(x)  # (B*X, 2)

        x = x.view(*size_cpy)  # B, C, H, W or B, C
        return x
