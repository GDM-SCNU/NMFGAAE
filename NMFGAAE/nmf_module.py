from dataloader import DataLoader
from metric import *
import torch

mse_loss = torch.nn.MSELoss()


def NMF(adj, k, epochs, U=None, V=None):
    if type(adj) != torch.Tensor:
        adj = torch.tensor(adj, dtype=torch.float)
    u = None
    v = None
    if U is None and V is None:
        u = torch.rand(adj.shape[0], k)
        v = torch.rand(k, adj.shape[1])
    else:
        u = U
        v = V
    for epoch in range(epochs):
        u = u * ((adj.matmul(v.t())) / (u.matmul(v).matmul(v.t()))) + 1e-20
        v = v * ((u.t().matmul(adj)) / (u.t().matmul(u).matmul(v))) + 1e-20
    return u, v


def NMF2(adj, k, epochs, U=None, V=None):
    if type(adj) != torch.Tensor:
        adj = torch.tensor(adj, dtype=torch.float)
    u = None
    v = None
    if U is None and V is None:
        u = torch.rand(adj.shape[0], k)
        v = torch.rand(k, adj.shape[1])
    else:
        u = U
        v = V
    for epoch in range(epochs):
        u = u * ((adj.matmul(v.t())) / (u.matmul(u.t()).matmul(adj).matmul(v.t()))) + 1e-20
        v = v * ((u.t().matmul(adj)) / (u.t().matmul(u).matmul(v))) + 1e-20
    return u, v
