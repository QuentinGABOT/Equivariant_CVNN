import torch
import torch.nn as nn
import torch.nn.functional as F


class Softmax(nn.Module):
    def __init__(self):
        super(Softmax, self).__init__()

    def forward(self, x, dim=1):
        x = F.softmax(x.type(torch.float64), dim=dim) # to ensure that the output is of type float64
        return x


class GumbelSoftmax(nn.Module):
    def __init__(self):
        super(GumbelSoftmax, self).__init__()

    def forward(self, x, tau, hard):
        x = F.gumbel_softmax(x, tau=tau, hard=hard)
        return x.type(torch.float64)  # to ensure that the output is of type float64


class SoftmaxMeanCtoR(nn.Module):
    def __init__(self):
        super(SoftmaxMeanCtoR, self).__init__()
        self.softmax = Softmax()

    def forward(self, x):
        return ((self.softmax(x.real) + self.softmax(x.imag)) / 2).type(
            torch.float64
        )  # to ensure that the output is of type float64


class GumbelSoftmaxMeanCtoR(nn.Module):
    def __init__(self):
        super(GumbelSoftmaxMeanCtoR, self).__init__()
        self.softmax = GumbelSoftmax()

    def forward(self, x, tau, hard):
        return (
            (self.softmax(x.real, tau, hard) + self.softmax(x.imag, tau, hard)) / 2
        ).type(
            torch.float64
        )  # to ensure that the output is of type float64


class SoftmaxProductCtoR(nn.Module):
    def __init__(self):
        super(SoftmaxProductCtoR, self).__init__()
        self.softmax = Softmax()

    def forward(self, x):
        return (self.softmax(self.softmax(x.real) * self.softmax(x.imag))).type(
            torch.float64
        )  # to ensure that the output is of type float64


class GumbelSoftmaxProductCtoR(nn.Module):
    def __init__(self):
        super(GumbelSoftmaxProductCtoR, self).__init__()
        self.softmax = GumbelSoftmax()

    def forward(self, x, tau, hard):
        return (
            self.softmax(
                self.softmax(x.real, tau, hard) * self.softmax(x.imag, tau, hard),
                tau,
                hard,
            )
        ).type(
            torch.float64
        )  # to ensure that the output is of type float64
