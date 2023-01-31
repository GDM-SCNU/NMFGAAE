import dgl
import torch
import pandas as pd
from dgl.data import citation_graph
import networkx as nx
import warnings

warnings.filterwarnings('ignore')


class DataLoader:
    """
    返回dgl.DGLGraph 和 labels
    """

    def add_self_loop(self, graph, adj):
        node_size = graph.number_of_nodes()
        for i in range(node_size):
            graph.add_edge(i, i)
            adj[i][i] = 1
        return graph, adj

    def remove_self_loop(self, graph, adj):
        node_size = graph.number_of_nodes()
        for i in range(node_size):
            if adj[i][i] == 1:
                graph.remove_edges([i, i])
                adj[i][i] = 0
        return graph, adj

    def graph_convert_to_adj(self, graph):
        """
        :param graph: networkx 中的DiGraph
        :return: 返回graph的邻接矩阵
        """
        edges = graph.edges  # 边信息
        node_size = graph.number_of_nodes()  # 节点个数
        adj = torch.zeros(node_size, node_size)  # 将要返回的邻接矩阵
        for i, j in edges:
            adj[i][j] = 1
        return adj

    def load_cora(self, path):
        """

        :param path: "../dataset/cora/cora"
        :return: graph, adj, label, k
        """
        cora_data = citation_graph.load_cora(path, verbose=False)
        graph = cora_data.graph  # networkx 的graph
        labels = torch.LongTensor(cora_data.labels)  # 节点对应的标签
        adj = self.graph_convert_to_adj(graph)  # graph对应的邻接矩阵
        graph = dgl.DGLGraph(graph)
        graph, adj = self.add_self_loop(graph, adj)
        return graph, adj, None, 7

    def load_citeseer(self, path):
        citeseer_data = citation_graph.load_citeseer(path, verbose=False)
        graph = citeseer_data.graph
        labels = torch.LongTensor(citeseer_data.labels)
        adj = self.graph_convert_to_adj(graph)
        graph = dgl.DGLGraph(graph)
        graph, adj = self.add_self_loop(graph, adj)
        return graph, adj, None, 6

    def load_karate_dataset(self, path):
        """

        :param path: "../dataset/karate"
        :return:
        """
        edges = pd.read_csv(path + '/edges.txt', sep='\t', header=None).to_numpy()
        labels = pd.read_csv(path + '/labels.txt', sep='\t', header=None).to_numpy()
        # labels = torch.LongTensor(labels)
        graph = nx.DiGraph()
        # 处理边， 构成graph
        for e in edges:
            graph.add_edge(e[0], e[1])
            graph.add_edge(e[1], e[0])

        # 处理labels， 转成LongTensor
        res_labels = []
        for labes in labels:
            res_labels.append(labes[1])

        labels = torch.LongTensor(res_labels)
        adj = self.graph_convert_to_adj(graph)
        graph = dgl.DGLGraph(graph)
        graph, adj = self.add_self_loop(graph, adj)
        return graph, adj, None, 2

    def load_webkb(self, path):
        edges = pd.read_csv(path + '/edges.txt', sep='\t', header=None).to_numpy()
        graph = nx.DiGraph()
        for e in edges:
            graph.add_edge(e[0], e[1])
            graph.add_edge(e[1], e[0])

        adj = self.graph_convert_to_adj(graph)
        graph = dgl.DGLGraph(graph)
        graph, adj = self.add_self_loop(graph, adj)
        return graph, adj, None, 4
        pass

    def load_syn_network(self, path=""):
        edges = pd.read_csv(path + '/network.dat', sep='\t', header=None).to_numpy()
        labels = pd.read_csv(path + '/community.dat', sep='\t', header=None).to_numpy()
        graph = nx.DiGraph()
        # 处理边， 构成graph
        for e in edges:
            graph.add_edge(e[0] - 1, e[1] - 1)
            graph.add_edge(e[1] - 1, e[0] - 1)

        # 处理labels， 转成LongTensor
        res_labels = []
        for labes in labels:
            res_labels.append(labes[1] - 1)

        labels = torch.LongTensor(res_labels)
        adj = self.graph_convert_to_adj(graph)
        graph = dgl.DGLGraph(graph)
        graph, adj = self.add_self_loop(graph, adj)
        k = self.num_of_community(path)
        return graph, adj, labels, k

    def num_of_community(self, path):
        community_labels = pd.read_csv(path + '/community.dat', sep='\t', header=None).to_numpy()
        community_set = set()
        for community in community_labels:
            community_set.add(community[1] - 1)
        return len(community_set)
        pass
