# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Code adapted from Heterogeneous Graph Transformer to extract graph structure
# from pre-processed OAG dataset
# https://github.com/acbull/pyHGT/blob/master/OAG/train_paper_field.py
# Copyright (c) 2019 acbull

import numpy as np
import pickle
from data import renamed_load

graph = renamed_load(open("graph_CS.pk", "rb"))
edge_list = graph.edge_list
edges = {}

field_name = "L1"

train_idx = []
val_idx = []
test_idx = []

# loop over all journal papers to create train, val, test split
cand_list = list(graph.edge_list["field"]["paper"]["PF_in_" + field_name].keys())
num_papers = len(graph.node_feature["paper"])
print(num_papers, len(cand_list))

labels = {}
for paper_id in graph.edge_list["paper"]["field"]["rev_PF_in_" + field_name]:
    label_list = []
    time_seen = None
    for field_id in graph.edge_list["paper"]["field"]["rev_PF_in_" + field_name][
        paper_id
    ]:
        _time = graph.edge_list["paper"]["field"]["rev_PF_in_" + field_name][paper_id][
            field_id
        ]
        assert _time is not None and (time_seen is None or _time == time_seen)
        time_seen = _time
        label_list.append(cand_list.index(field_id))
    assert time_seen is not None
    if time_seen < 2015:
        train_idx.append(paper_id)
    elif time_seen <= 2016:
        val_idx.append(paper_id)
    else:
        test_idx.append(paper_id)
    labels[paper_id] = np.array(label_list)

node_types = set()

for dtype in edge_list:
    for stype in edge_list[dtype]:
        for rel in edge_list[dtype][stype]:
            if rel != "PF_in_" + field_name and not rel.startswith("rev_"):
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

pickle.dump(graph, open("graph_{}.pk".format(field_name), "wb"))
