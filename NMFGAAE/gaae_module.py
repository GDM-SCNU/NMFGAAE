import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_community):
        super(GATLayer, self).__init__()
        self.g = g
        # 公式 (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # 公式 (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.attn_fc2 = nn.Linear(2 * num_community, 1, bias=False)

    def edge_attention(self, edges):
        # 公式 (2) 所需，边上的用户定义函数
        # z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        z2 = torch.cat([edges.src['u'], edges.dst['u']], dim=1)
        a = self.attn_fc2(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # 公式 (3), (4)所需，传递消息用的用户定义函数
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # 公式 (3), (4)所需, 归约用的用户定义函数
        # 公式 (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # 公式 (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, h, u):
        # 公式 (1)
        z = self.fc(h)
        self.g.ndata['z'] = z
        self.g.ndata['u'] = u  # 用来计算注意力的矩阵
        # 公式 (2)
        self.g.apply_edges(self.edge_attention)
        # 公式 (3) & (4)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')


class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads,num_community, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim, num_community))
        self.merge = merge

    def forward(self, h, u):
        head_outs = [attn_head(h, u) for attn_head in self.heads]
        if self.merge == 'cat':
            # 对输出特征维度（第1维）做拼接
            return torch.cat(head_outs, dim=1)
        else:
            # 用求平均整合多头结果
            return torch.mean(torch.stack(head_outs))


class GAAE(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads, numcommunity):
        super(GAAE, self).__init__()
        self.layer1 = MultiHeadGATLayer(g, in_dim, hidden_dim, num_heads, num_community=numcommunity)
        # 注意输入的维度是 hidden_dim * num_heads 因为多头的结果都被拼接在了
        # 一起。 此外输出层只有一个头。
        self.layer2 = MultiHeadGATLayer(g, hidden_dim * num_heads, out_dim, num_heads=1, num_community=numcommunity)

    def forward(self, h, u):
        h = self.layer1(h, u)
        h = F.elu(h)
        h = self.layer2(h, u)
        # h = torch.sigmoid(h)

        return h, torch.sigmoid(h.matmul(h.t()))
