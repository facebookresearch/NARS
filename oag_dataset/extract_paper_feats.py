# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from data import renamed_load

graph = renamed_load(open("graph_CS.pk", "rb"))
ntype = "paper"
feature = np.array(list(graph.node_feature[ntype]["emb"]))
with open(f"{ntype}.npy", "wb") as f:
    np.save(f, feature)
