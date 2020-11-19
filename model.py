# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


class FeedForwardNet(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, n_layers, dropout):
        super(FeedForwardNet, self).__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        if n_layers == 1:
            self.layers.append(nn.Linear(in_feats, out_feats))
        else:
            self.layers.append(nn.Linear(in_feats, hidden))
            for i in range(n_layers - 2):
                self.layers.append(nn.Linear(hidden, hidden))
            self.layers.append(nn.Linear(hidden, out_feats))
        if self.n_layers > 1:
            self.prelu = nn.PReLU()
            self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer_id, layer in enumerate(self.layers):
            x = layer(x)
            if layer_id < self.n_layers - 1:
                x = self.dropout(self.prelu(x))
        return x


class SIGN(nn.Module):
    def __init__(
        self, in_feats, hidden, out_feats, num_hops, n_layers, dropout, input_drop
    ):
        super(SIGN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.prelu = nn.PReLU()
        self.inception_ffs = nn.ModuleList()
        self.input_drop = input_drop
        for i in range(num_hops):
            self.inception_ffs.append(
                FeedForwardNet(in_feats, hidden, hidden, n_layers, dropout)
            )
        self.project = FeedForwardNet(
            num_hops * hidden, hidden, out_feats, n_layers, dropout
        )

    def forward(self, feats):
        hidden = []
        for feat, ff in zip(feats, self.inception_ffs):
            if self.input_drop:
                feat = self.dropout(feat)
            hidden.append(ff(feat))
        out = self.project(self.dropout(self.prelu(torch.cat(hidden, dim=-1))))
        return torch.log_softmax(out, dim=-1)


class WeightedAggregator(nn.Module):
    def __init__(self, num_feats, in_feats, num_hops):
        super(WeightedAggregator, self).__init__()
        self.agg_feats = nn.ParameterList()
        for _ in range(num_hops):
            self.agg_feats.append(nn.Parameter(torch.Tensor(num_feats, in_feats)))
            nn.init.xavier_uniform_(self.agg_feats[-1])

    def forward(self, feats):
        new_feats = []
        for feat, weight in zip(feats, self.agg_feats):
            new_feats.append((feat * weight.unsqueeze(0)).sum(dim=1).squeeze())
        return new_feats


class PartialWeightedAggregator(nn.Module):
    def __init__(self, num_feats, in_feats, num_hops, sample_size):
        super(PartialWeightedAggregator, self).__init__()
        self.weight_store = []
        self.agg_feats = nn.ParameterList()
        self.discounts = nn.ParameterList()
        self.num_hops = num_hops
        for _ in range(num_hops):
            self.weight_store.append(torch.Tensor(num_feats, in_feats))
            self.agg_feats.append(nn.Parameter(torch.Tensor(sample_size, in_feats)))
            self.discounts.append(nn.Parameter(torch.Tensor(in_feats)))
            nn.init.xavier_uniform_(self.weight_store[-1])
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_hops):
            nn.init.zeros_(self.agg_feats[i])
            nn.init.ones_(self.discounts[i])

    def update_selected(self, selected):
        for param, weight, discount in zip(
            self.agg_feats, self.weight_store, self.discounts
        ):
            weight *= discount
            weight[selected] += param.data
        self.reset_parameters()

    def forward(self, args):
        feats, old_sum = args
        new_feats = []
        for feat, weight, old_feat, discount in zip(
            feats, self.agg_feats, old_sum, self.discounts
        ):
            new_feats.append(
                (feat * weight.unsqueeze(0)).sum(dim=1).squeeze() + old_feat * discount
            )
        return new_feats
