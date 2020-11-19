# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Code adapted from Heterogeneous Graph Transformer to extract graph structure
# from pre-processed OAG dataset
# https://github.com/acbull/pyHGT/blob/master/OAG/train_paper_venue.py
# Copyright (c) 2019 acbull


import numpy as np
import pickle
from data import renamed_load

graph = renamed_load(open("graph_CS.pk", "rb"))
edge_list = graph.edge_list
edges = {}


train_idx = []
train_label = []
val_idx = []
val_label = []
test_idx = []
test_label = []

# loop over all journal papers to create train, val, test split
cand_list = list(graph.edge_list["venue"]["paper"]["PV_Journal"].keys())

for paper_id in graph.edge_list["paper"]["venue"]["rev_PV_Journal"]:
    for venue_id in graph.edge_list["paper"]["venue"]["rev_PV_Journal"][paper_id]:
        _time = graph.edge_list["paper"]["venue"]["rev_PV_Journal"][paper_id][venue_id]
        assert _time is not None
        if _time < 2015:
            if paper_id not in train_idx:
                train_idx.append(paper_id)
                train_label.append(cand_list.index(venue_id))
        elif _time <= 2016:
            if paper_id not in val_idx:
                val_idx.append(paper_id)
                val_label.append(cand_list.index(venue_id))
        else:
            if paper_id not in test_idx:
                test_idx.append(paper_id)
                test_label.append(cand_list.index(venue_id))

num_papers = len(graph.node_feature["paper"])
labels = np.zeros(num_papers, dtype=np.long) - 1  # init to -1
labels[train_idx] = train_label
labels[val_idx] = val_label
labels[test_idx] = test_label

node_types = set()

for dtype in edge_list:
    for stype in edge_list[dtype]:
        for rel in edge_list[dtype][stype]:
            if rel != "PV_Journal" and not rel.startswith("rev_"):
                print(stype, dtype, rel)
                src_list = []
                dst_list = []
                for dst in edge_list[dtype][stype][rel]:
                    for src in edge_list[dtype][stype][rel][dst]:
                        src_list.append(src)
                        dst_list.append(dst)
                src_list = np.array(src_list)
                dst_list = np.array(dst_list)
                edges[(stype, rel, dtype)] = (src_list, dst_list)
                node_types.add(stype)
                node_types.add(dtype)

n_classes = len(cand_list)

graph = {
    "edges": edges,
    "labels": labels,
    "split": [train_idx, val_idx, test_idx],
    "n_classes": n_classes,
}

pickle.dump(graph, open("graph_venue.pk", "wb"))
