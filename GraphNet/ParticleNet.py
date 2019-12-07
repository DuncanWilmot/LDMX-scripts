from __future__ import print_function

import dgl
import dgl.function as fn
from dgl.transform import remove_self_loop
import torch.nn as nn
from dgl_utils import segmented_knn_graph


class EdgeConvBlock(nn.Module):
    r"""EdgeConv layer.
    Introduced in "`Dynamic Graph CNN for Learning on Point Clouds
    <https://arxiv.org/pdf/1801.07829>`__".  Can be described as follows:
    .. math::
       x_i^{(l+1)} = \max_{j \in \mathcal{N}(i)} \mathrm{ReLU}(
       \Theta \cdot (x_j^{(l)} - x_i^{(l)}) + \Phi \cdot x_i^{(l)})
    where :math:`\mathcal{N}(i)` is the neighbor of :math:`i`.
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    batch_norm : bool
        Whether to include batch normalization on messages.
    """

    def __init__(self, in_feat, out_feats, batch_norm=True, activation=True):
        super(EdgeConvBlock, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.num_layers = len(out_feats)

        out_feat = out_feats[0]
        self.theta = nn.Linear(in_feat, out_feat, bias=False if self.batch_norm else True)
        self.phi = nn.Linear(in_feat, out_feat, bias=False if self.batch_norm else True)
        self.fcs = nn.ModuleList()
        for i in range(1, self.num_layers):
            self.fcs.append(nn.Linear(out_feats[i - 1], out_feats[i], bias=False if self.batch_norm else True))

        if batch_norm:
            self.bns = nn.ModuleList()
            for i in range(self.num_layers):
                self.bns.append(nn.BatchNorm1d(out_feats[i]))

        if activation:
            self.acts = nn.ModuleList()
            for i in range(self.num_layers):
                self.acts.append(nn.ReLU())

        if in_feat == out_feats[-1]:
            self.sc = None
        else:
            self.sc = nn.Linear(in_feat, out_feats[-1], bias=False if self.batch_norm else True)
            self.sc_bn = nn.BatchNorm1d(out_feats[-1])

        if activation:
            self.sc_act = nn.ReLU()

    def message(self, edges):
        """The message computation function.
        """
        theta_x = self.theta(edges.dst['x'] - edges.src['x'])
        phi_x = self.phi(edges.src['x'])
        return {'e': theta_x + phi_x}

    def forward(self, g, h):
        """Forward computation
        Parameters
        ----------
        g : DGLGraph
            The graph.
        h : Tensor
            :math:`(N, D)` where :math:`N` is the number of nodes and
            :math:`D` is the number of feature dimensions.
        Returns
        -------
        torch.Tensor
            New node features.
        """
        with g.local_scope():
            g.ndata['x'] = h
            # generate the message and store it on the edges
            g.apply_edges(self.message)
            # process the message
            e = g.edata['e']
            for i in range(self.num_layers):
                if i > 0:
                    e = self.fcs[i - 1](e)
                if self.batch_norm:
                    e = self.bns[i](e)
                if self.activation:
                    e = self.acts[i](e)
            g.edata['e'] = e
            # pass the message and update the nodes
            g.update_all(fn.copy_e('e', 'e'), fn.mean('e', 'x'))
            # shortcut connection
            x = g.ndata.pop('x')
            g.edata.pop('e')
            if self.sc is None:
                sc = h
            else:
                sc = self.sc(h)
                if self.batch_norm:
                    sc = self.sc_bn(sc)
            if self.activation:
                return self.sc_act(x + sc)
            else:
                return x + sc


class ParticleNet(nn.Module):

    def __init__(self,
                 input_dims,
                 num_classes,
                 conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))],
                 fc_params=[(128, 0.1)],
                 **kwargs):
        super(ParticleNet, self).__init__(**kwargs)

        self.bn_fts = nn.BatchNorm1d(input_dims)

        self.k_neighbors = []
        self.edge_convs = nn.ModuleList()
        for idx, layer_param in enumerate(conv_params):
            k, channels = layer_param
            in_feat = input_dims if idx == 0 else conv_params[idx - 1][1][-1]
            self.edge_convs.append(EdgeConvBlock(in_feat=in_feat, out_feats=channels))
            self.k_neighbors.append(k)

        fcs = []
        for idx, layer_param in enumerate(fc_params):
            channels, drop_rate = layer_param
            if idx == 0:
                in_chn = conv_params[-1][1][-1]
            else:
                in_chn = fc_params[idx - 1][0]
            fcs.append(nn.Sequential(nn.Linear(in_chn, channels), nn.ReLU(), nn.Dropout(drop_rate)))
        fcs.append(nn.Linear(fc_params[-1][0], num_classes))
        self.fc = nn.Sequential(*fcs)


    def forward(self, batch_graph, features):
        g = batch_graph
        fts = self.bn_fts(features)
        for idx, (k, conv) in enumerate(zip(self.k_neighbors, self.edge_convs)):
            if idx > 0:
                g = remove_self_loop(segmented_knn_graph(fts, k + 1, batch_graph.batch_num_nodes))
            fts = conv(g, fts)

        batch_graph.ndata['fts'] = fts
        x = dgl.mean_nodes(batch_graph, 'fts')
        return self.fc(x)
