import torch
import torch.nn as nn
import torch.nn.functional as F
import torchcvnn.nn.modules as c_nn


class PolyCtoR(nn.Module):
    def __init__(self, order=3):
        super(PolyCtoR, self).__init__()
        self.order = order
        size = ((self.order + 1) * (self.order + 2)) // 2 - 1
        self.poly = nn.Linear(size, 1, dtype=torch.float64, bias=True)

    def forward(self, x):
        size_cpy = x.size()  # B, C, H, W or B, C
        x = torch.view_as_real(x).type(torch.float64)  # B, C, H, W, 2 or B, C, 2

        x = x.reshape(-1, 2)  # (B*X, 2)

        x_real = x[:, 0]
        x_imag = x[:, 1]

        # Calculate the binomial expansion
        combinations = list(binomial_expansion(x_real, x_imag, self.order).values())

        # Stack them along a new dimension (N dimension)
        x = torch.stack(combinations, dim=1)

        x = self.poly(x)  # (B*X, 2*self.order) -> (B*X, 1)

        x = x.view(*size_cpy)  # B, C, H, W or B, C
        return x.type(torch.float64)  # to ensure that the output is of type float64


def binomial_expansion(a, b, N):
    expansion_terms = {}  # Start with the constant term 1

    for order in range(1, N + 1):
        # Loop through all combinations of i and j such that i + j = order
        for i in range(order + 1):
            j = order - i

            # Create the LaTeX-style key representing the term
            a_part = f"a^{i}" if i > 0 else ""
            b_part = f"b^{j}" if j > 0 else ""

            if i > 1:
                a_part = f"a^{i}"
            elif i == 1:
                a_part = "a"
            else:
                a_part = ""

            if j > 1:
                b_part = f"b^{j}"
            elif j == 1:
                b_part = "b"
            else:
                b_part = ""

            key = f"${a_part}{b_part}$"

            # Calculate term a^i * b^j
            term = torch.pow(a, i) * torch.pow(b, j)

            # Store the term in the dictionary
            expansion_terms[key] = term

    return expansion_terms


class MLPCtoR(nn.Module):
    def __init__(self):
        super(MLPCtoR, self).__init__()
        self.linear1 = nn.Linear(2, 16, dtype=torch.float64, bias=True)
        self.linear2 = nn.Linear(16, 32, dtype=torch.float64, bias=True)
        self.linear3 = nn.Linear(32, 1, dtype=torch.float64, bias=True)
        self.relu = nn.ReLU()

        self.mlp = nn.Sequential(
            self.linear1,
            self.relu,
            self.linear2,
            self.relu,
            self.linear3,
        )

    def forward(self, x):
        size_cpy = x.size()  # B, C, H, W or B, C
        x = torch.view_as_real(x).type(torch.float64)  # B, C, H, W, 2 or B, C, 2
        if x.dim() not in [3, 5]:
            raise ValueError("Unexpected tensor shape")

        x = x.reshape(-1, 2)  # (B*X, 2)

        x = self.mlp(x)  # (B*X, 2)

        x = x.view(*size_cpy)  # B, C, H, W or B, C
        return x.type(torch.float64)  # to ensure that the output is of type float64


class ModCtoR(nn.Module):
    def __init__(self):
        super(ModCtoR, self).__init__()
        self.mod = c_nn.Mod()

    def forward(self, x):
        x = self.mod(x)
        return x.type(torch.float64)  # to ensure that the output is of type float64


class NoCtoR(nn.Module):
    def __init__(self):
        super(NoCtoR, self).__init__()

    def forward(self, x):
        return x
