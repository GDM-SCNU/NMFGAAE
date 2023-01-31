import warnings
from metric import *
from nmf_module import NMF, NMF2
from dataloader import DataLoader
from gaae_module import GAAE
from loss_func import loss_fun
import torch

warnings.filterwarnings('ignore')


def NMFGAAE(graph, A, labels, k, max_iter):
    _, X = NMF(adj=A, k=64, epochs=200)  # 可以使用单位阵、或任何嵌入方法来初始化
    X = X.t()

    n_in_feat = X.shape[1]  # gcn输入特征的维度
    n_hidden = 2048  # gcn隐藏层的维度
    n_out_feat = 1024  # gcn学习到的表示的维度

    lr = 1e-4  # gcn学习率
    alpha = 1  # loss中属于nmf那一项的系数
    num_header = 1
    weight_decay = 3e-4

    model = GAAE(graph, n_in_feat, n_hidden, n_out_feat, num_heads=num_header, numcommunity=k)
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)

    u = torch.rand(graph.number_of_nodes(), k)
    v = torch.rand(k, n_out_feat)

    for epoch in range(max_iter):

        Z, A1 = model(X, u.detach())

        Z = torch.sigmoid(Z)
        u, v = NMF2(Z, k, 200, u.detach(), v.detach())
        Z1 = torch.sigmoid(u.matmul(v))
        loss = loss_fun(A, A1.detach(), Z, Z1.detach(), alpha=alpha)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = torch.softmax(u, dim=1).argmax(dim=1).detach().numpy()
        ac, nmi, ari = 0., 0., 0.
        if labels is not None:
            ac = AC(pred, labels)
            nmi = NMI(pred, labels)
            ari = ARI(pred, labels)
        Q = Modularity(A, pred)
        if labels is not None:
            print('epoch={:d}, Q={:.3f}, AC={:.3F}, NMI={:.3f}, ARI={:.3f}'.format(epoch, Q, ac, nmi, ari))
        else:
            print('epoch={:d}, Q={:.3f}'.format(epoch, Q))


def example1():
    dataloader = DataLoader()
    g, A, lables, k = dataloader.load_cora("./dataset/cora")
    NMFGAAE(g, A, lables, k, 50)
    pass


def example2():
    dataloader = DataLoader()
    g, A, lables, k = dataloader.load_webkb("./dataset/webkb")
    NMFGAAE(g, A, lables, k, 50)
    pass


def example3():
    dataloader = DataLoader()
    g, A, lables, k = dataloader.load_syn_network("./dataset/syn_network/gn2ku02")
    NMFGAAE(g, A, lables, k, 50)
    pass


if __name__ == '__main__':
    example3()
