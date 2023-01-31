import torch

bce_loss = torch.nn.BCELoss(size_average=False)


def loss_fun(A, A1, Z, Z1, alpha):
    return bce_loss(A.view(-1), A1.view(-1)) + alpha * bce_loss(Z.view(-1), Z1.view(-1))



