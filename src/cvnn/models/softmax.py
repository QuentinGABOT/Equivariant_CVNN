import torch
import torch.nn as nn
import torch.nn.functional as F


class GumbelSoftmax(nn.Module):
    def __init__(self):
        super(GumbelSoftmax, self).__init__()

    def forward(self, x, tau, hard):
        return F.gumbel_softmax(x, tau=tau, hard=hard)


class GumbelSoftmaxMeanCtoR(nn.Module):
    def __init__(self):
        super(GumbelSoftmaxMeanCtoR, self).__init__()
        self.gumbel_softmax = GumbelSoftmax()

    def forward(self, x, tau, hard):
        return (
            self.gumbel_softmax(x.real, tau, hard)
            + self.gumbel_softmax(x.imag, tau, hard)
        ) / 2


class GumbelSoftmaxProductCtoR(nn.Module):
    def __init__(self):
        super(GumbelSoftmaxProductCtoR, self).__init__()
        self.gumbel_softmax = GumbelSoftmax()

    def forward(self, x, tau, hard):
        return self.gumbel_softmax(
            self.gumbel_softmax(x.real, tau, hard)
            * self.gumbel_softmax(x.imag, tau, hard),
            tau,
            hard,
        )


class Softmax(nn.Module):
    def __init__(self):
        super(Softmax, self).__init__()

    def forward(self, x, dim=1):
        return F.softmax(x, dim=dim)


class SoftmaxMeanCtoR(nn.Module):
    def __init__(self):
        super(SoftmaxMeanCtoR, self).__init__()
        self.softmax = Softmax()

    def forward(self, x, dim=1):
        return (self.softmax(x.real, dim) + self.softmax(x.imag, dim)) / 2


class SoftmaxProductCtoR(nn.Module):
    def __init__(self):
        super(SoftmaxProductCtoR, self).__init__()
        self.softmax = Softmax()

    def forward(self, x, dim=1):
        return self.softmax(self.softmax(x.real, dim) * self.softmax(x.imag, dim), dim)
