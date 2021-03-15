import torch
import torch.nn as nn
import dgl.nn.pytorch as dglnn
import dgl
import tqdm
import layers
from data_utils import *
import dgl.nn as dglnn


class binaryRGCN(nn.Module):
    """
    RGCN with binary label on the specified entity
    """
    def __init__(self, in_dim, h_dim, n_layers, activation, dropout, rel_names, label_entity):
        super().__init__()
        self.h_dim = h_dim
        self.in_dim = in_dim
        self.layers = nn.ModuleList()
        # self.batch_norms = nn.ModuleList()
        #i2h
        self.layers.append(dglnn.HeteroGraphConv(
            {rel: dglnn.SAGEConv(in_dim, h_dim, 'mean', activation=activation) for rel in rel_names}))
        # self.batch_norms.append(nn.BatchNorm1d(h_dim))
        #h2h
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.HeteroGraphConv(
                {rel: dglnn.SAGEConv(h_dim, h_dim, 'mean', feat_drop=dropout, activation=activation) for rel in rel_names}))
            # self.batch_norms.append(nn.BatchNorm1d(h_dim))
        #h2o
        self.layers.append(dglnn.HeteroGraphConv(
            {rel: dglnn.SAGEConv(h_dim, 1, 'mean', feat_drop=dropout) for rel in rel_names}))
        self.label_entity = label_entity

    def forward(self, blocks, h):
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h_dst = {k: v[:block.num_dst_nodes(k)] for k, v in h.items()}
            h = layer(block, (h, h_dst))
        h = torch.sigmoid(h[self.label_entity])
        return h

    def inference(self, g, x, device, batch_size, num_workers, is_pad):
        for l, layer in enumerate(self.layers):
            y = {
                k: torch.zeros(
                    g.number_of_nodes(k),
                    self.h_dim if l != len(self.layers) - 1 else 1)
                for k in g.ntypes}

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                {k: torch.arange(g.number_of_nodes(k)) for k in g.ntypes},
                sampler,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=num_workers)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0].int().to(device)
                is_pad = (l == 0) and is_pad
                h = load_feature_subtensor(x, input_nodes, is_pad, device)
                h_dst = {k: v[:block.num_dst_nodes(k)] for k, v in h.items()}
                h = layer(block, (h, h_dst))
                for k in h.keys():
                    y[k][output_nodes[k]] = h[k].cpu()
            x = y
        y = torch.sigmoid(y[self.label_entity])
        return y


class RGCN(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, n_layers, activation, dropout, rel_names):
        super().__init__()
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.layers = nn.ModuleList()
        #i2h
        self.layers.append(dglnn.HeteroGraphConv(
            {rel: dglnn.GraphConv(in_dim, h_dim) for rel in rel_names}))
        #h2h
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.HeteroGraphConv(
                {rel: dglnn.GraphConv(h_dim, h_dim) for rel in rel_names}))
        #h2o
        self.layers.append(dglnn.HeteroGraphConv(
            {rel: dglnn.GraphConv(h_dim, out_dim) for rel in rel_names}))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, h):
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h_dst = {k: v[:block.num_dst_nodes(k)] for k, v in h.items()}
            h = layer(block, (h, h_dst))
            if l != len(self.layers) - 1:
                h = {k: self.activation(v) for k, v in h.items()}
                h = {k: self.dropout(v) for k, v in h.items()}
        return h

    def inference(self, g, x, device, batch_size, num_workers, is_pad):
        for l, layer in enumerate(self.layers):
            y = {
                k: torch.zeros(
                    g.number_of_nodes(k),
                    self.h_dim if l != len(self.layers) - 1 else self.out_dim)
                for k in g.ntypes}

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                {k: torch.arange(g.number_of_nodes(k)) for k in g.ntypes},
                sampler,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=num_workers)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0].int().to(device)
                is_pad = (l == 0) and is_pad
                h = load_feature_subtensor(x, input_nodes, is_pad, device)
                h_dst = {k: v[:block.num_dst_nodes(k)] for k, v in h.items()}
                h = layer(block, (h, h_dst))
                if l != len(self.layers) - 1:
                    h = {k: self.activation(v) for k, v in h.items()}
                    h = {k: self.dropout(v) for k, v in h.items()}

                for k in h.keys():
                    y[k][output_nodes[k]] = h[k].cpu()

            x = y
        return y


class HeteroRGCN(nn.Module):
    """
    inefficient full batch(graph) forward, applicable only to small graphs
    """
    def __init__(self, G, in_size, hidden_size, out_size):
        super(HeteroRGCN, self).__init__()
        # Use trainable node embeddings as featureless inputs.
        embed_dict = {ntype : nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), in_size))
                      for ntype in G.ntypes}
        for key, embed in embed_dict.items():
            nn.init.xavier_uniform_(embed)
        self.embed = nn.ParameterDict(embed_dict)
        # create layers
        self.layer1 = layers.HeteroRGCNLayer(in_size, hidden_size, G.etypes)
        self.layer2 = layers.HeteroRGCNLayer(hidden_size, out_size, G.etypes)

    def forward(self, G):
        h_dict = self.layer1(G, self.embed)
        h_dict = {k : F.leaky_relu(h) for k, h in h_dict.items()}
        h_dict = self.layer2(G, h_dict)
        return h_dict

# # chuixue's version with fixed input feature, target node init to input_feature, other type of node to 0
# class HeteroRGCN(nn.Module):
#     def __init__(self, graph, target_node, node_feature, in_feats, h_dim, num_classes=2):
#         super(HeteroRGCN, self).__init__()
#
#
#         embed_dict = {ntype: torch.Tensor(graph.number_of_nodes(ntype), in_feats).to(device) for ntype in graph.ntypes}
#
#         for key, embed in embed_dict.items():
#             embed_dict[key] = nn.init.zeros_(embed)
#
#         #    embed_dict['user'] = nn.Parameter(graph.nodes['user'].data['f'].float())
#
#         self.embed = embed_dict
#         self.embed[target_node] = node_feature
#
#         self.layer1 = layers.HeteroRGCNLayer(in_feats, h_dim, graph.etypes)
#         self.layer2 = layers.HeteroRGCNLayer(h_dim, num_classes, graph.etypes)
#
#     def forward(self, graph):
#         h_dict = self.layer1(graph, self.embed)
#         h_dict = {k: F.leaky_relu(h) for k, h in h_dict.items()}
#         h_dict = self.layer2(graph, h_dict)
#         return h_dict
