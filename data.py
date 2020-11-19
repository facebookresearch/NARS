# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import torch
import dgl
import dgl.function as fn


###############################################################################
# Loading Relation Subsets
###############################################################################

def read_relation_subsets(fname):
    print("Reading Relation Subsets:")
    rel_subsets = []
    with open(fname) as f:
        for line in f:
            relations = line.strip().split(',')
            rel_subsets.append(relations)
            print(relations)
    return rel_subsets


###############################################################################
# Generate multi-hop neighbor-averaged feature for each relation subset
###############################################################################

def gen_rel_subset_feature(g, rel_subset, args, device):
    """
    Build relation subgraph given relation subset and generate multi-hop
    neighbor-averaged feature on this subgraph
    """
    if args.cpu_preprocess:
        device = "cpu"
    new_edges = {}
    ntypes = set()
    for etype in rel_subset:
        stype, _, dtype = g.to_canonical_etype(etype)
        src, dst = g.all_edges(etype=etype)
        src = src.numpy()
        dst = dst.numpy()
        new_edges[(stype, etype, dtype)] = (src, dst)
        new_edges[(dtype, etype + "_r", stype)] = (dst, src)
        ntypes.add(stype)
        ntypes.add(dtype)
    new_g = dgl.heterograph(new_edges)

    # set node feature and calc deg
    for ntype in ntypes:
        num_nodes = new_g.number_of_nodes(ntype)
        if num_nodes < g.nodes[ntype].data["feat"].shape[0]:
            new_g.nodes[ntype].data["hop_0"] = g.nodes[ntype].data["feat"][:num_nodes, :]
        else:
            new_g.nodes[ntype].data["hop_0"] = g.nodes[ntype].data["feat"]
        deg = 0
        for etype in new_g.etypes:
            _, _, dtype = new_g.to_canonical_etype(etype)
            if ntype == dtype:
                deg = deg + new_g.in_degrees(etype=etype)
        norm = 1.0 / deg.float()
        norm[torch.isinf(norm)] = 0
        new_g.nodes[ntype].data["norm"] = norm.view(-1, 1).to(device)

    res = []

    # compute k-hop feature
    for hop in range(1, args.R + 1):
        ntype2feat = {}
        for etype in new_g.etypes:
            stype, _, dtype = new_g.to_canonical_etype(etype)
            new_g[etype].update_all(fn.copy_u(f'hop_{hop-1}', 'm'), fn.sum('m', 'new_feat'))
            new_feat = new_g.nodes[dtype].data.pop("new_feat")
            assert("new_feat" not in new_g.nodes[stype].data)
            if dtype in ntype2feat:
                ntype2feat[dtype] += new_feat
            else:
                ntype2feat[dtype] = new_feat
        for ntype in new_g.ntypes:
            assert ntype in ntype2feat  # because subgraph is not directional
            feat_dict = new_g.nodes[ntype].data
            old_feat = feat_dict.pop(f"hop_{hop-1}")
            if ntype == "paper":
                res.append(old_feat.cpu())
            feat_dict[f"hop_{hop}"] = ntype2feat.pop(ntype).mul_(feat_dict["norm"])

    res.append(new_g.nodes["paper"].data.pop(f"hop_{args.R}").cpu())
    return res


###############################################################################
# Dataset (ACM, MAG, OAG) loading
###############################################################################


def load_data(device, args):
    if args.cpu_preprocess:
        device = "cpu"
    with torch.no_grad():
        if args.dataset.startswith("acm"):
            return load_acm(device, args)
        elif args.dataset == "mag":
            return load_mag(device, args)
        elif args.dataset.startswith("oag"):
            return load_oag(device, args)
        else:
            raise RuntimeError(f"Dataset {args.dataset} not supported")


def load_acm(device, args):
    g, labels, n_classes, train_nid, val_nid, test_nid = load_acm_raw()

    features = g.nodes["paper"].data["feat"]

    path = args.use_emb
    author_emb = torch.load(os.path.join(path, "author.pt")).float()
    field_emb = torch.load(os.path.join(path, "field.pt")).float()

    g.nodes["author"].data["feat"] = author_emb.to(device)
    g.nodes["field"].data["feat"] = field_emb.to(device)
    g.nodes["paper"].data["feat"] = features.to(device)
    paper_dim = g.nodes["paper"].data["feat"].shape[1]
    author_dim = g.nodes["author"].data["feat"].shape[1]
    assert(paper_dim >= author_dim)
    if paper_dim > author_dim:
        print(f"Randomly embedding features from dimension {author_dim} to {paper_dim}")
        author_feat = g.nodes["author"].data.pop("feat")
        field_feat = g.nodes["field"].data.pop("feat")
        rand_weight = torch.Tensor(author_dim, paper_dim).uniform_(-0.5, 0.5).to(device)
        g.nodes["author"].data["feat"] = torch.matmul(author_feat, rand_weight)
        g.nodes["field"].data["feat"] = torch.matmul(field_feat, rand_weight)

    labels = labels.to(device)
    train_nid, val_nid, test_nid = np.array(train_nid), np.array(val_nid), np.array(test_nid)

    return g, labels, n_classes, train_nid, val_nid, test_nid


def load_mag(device, args):
    from ogb.nodeproppred import DglNodePropPredDataset
    path = args.use_emb
    home_dir = os.getenv("HOME")
    dataset = DglNodePropPredDataset(
        name="ogbn-mag", root=os.path.join(home_dir, ".ogb", "dataset"))
    g, labels = dataset[0]
    splitted_idx = dataset.get_idx_split()
    train_nid = splitted_idx["train"]['paper']
    val_nid = splitted_idx["valid"]['paper']
    test_nid = splitted_idx["test"]['paper']
    features = g.nodes['paper'].data['feat']
    author_emb = torch.load(os.path.join(path, "author.pt")).float()
    topic_emb = torch.load(os.path.join(path, "field_of_study.pt")).float()
    institution_emb = torch.load(os.path.join(path, "institution.pt")).float()

    g.nodes["author"].data["feat"] = author_emb.to(device)
    g.nodes["institution"].data["feat"] = institution_emb.to(device)
    g.nodes["field_of_study"].data["feat"] = topic_emb.to(device)
    g.nodes["paper"].data["feat"] = features.to(device)
    paper_dim = g.nodes["paper"].data["feat"].shape[1]
    author_dim = g.nodes["author"].data["feat"].shape[1]
    if paper_dim != author_dim:
        paper_feat = g.nodes["paper"].data.pop("feat")
        rand_weight = torch.Tensor(paper_dim, author_dim).uniform_(-0.5, 0.5)
        g.nodes["paper"].data["feat"] = torch.matmul(paper_feat, rand_weight.to(device))
        print(f"Randomly project paper feature from dimension {paper_dim} to {author_dim}")

    labels = labels['paper'].to(device).squeeze()
    n_classes = int(labels.max() - labels.min()) + 1
    train_nid, val_nid, test_nid = np.array(train_nid), np.array(val_nid), np.array(test_nid)
    return g, labels, n_classes, train_nid, val_nid, test_nid


def load_oag(device, args):
    import pickle
    assert args.data_dir is not None
    if args.dataset == "oag_L1":
        graph_file = "graph_L1.pk"
        predict_venue = False
    elif args.dataset == "oag_venue":
        graph_file = "graph_venue.pk"
        predict_venue = True
    else:
        raise RuntimeError(f"Unsupported dataset {args.dataset}")
    with open(os.path.join(args.data_dir, graph_file), "rb") as f:
        dataset = pickle.load(f)
    n_classes = dataset["n_classes"]
    graph = dgl.heterograph(dataset["edges"])
    train_nid, val_nid, test_nid = dataset["split"]

    # use relational embedding that we generate
    path = args.use_emb
    author_emb = torch.load(os.path.join(path, "author.pt")).float().to(device)
    field_emb = torch.load(os.path.join(path, "field.pt")).float().to(device)
    venue_emb = torch.load(os.path.join(path, "venue.pt")).float().to(device)
    affiliation_emb = torch.load(os.path.join(path, "affiliation.pt")).float().to(device)
    with open(os.path.join(args.data_dir, "paper.npy"), "rb") as f:
        # loading lang features of paper provided by HGT author
        paper_feat = torch.from_numpy(np.load(f)).float().to(device)
    author_dim = author_emb.shape[1]
    paper_dim = paper_feat.shape[1]
    if author_dim < paper_dim:
        print(f"Randomly project paper feature from dimension {author_dim} to {paper_dim}")
        rand_weight = torch.Tensor(author_dim, paper_dim).uniform_(-0.5, 0.5).to(device)
        author_emb = torch.matmul(author_emb, rand_weight)
        field_emb = torch.matmul(field_emb, rand_weight)
        venue_emb = torch.matmul(venue_emb, rand_weight)
        affiliation_emb = torch.matmul(affiliation_emb, rand_weight)
    graph.nodes["paper"].data["feat"] = paper_feat[:graph.number_of_nodes("paper")]
    graph.nodes["author"].data["feat"] = author_emb[:graph.number_of_nodes("author")]
    graph.nodes["affiliation"].data["feat"] = affiliation_emb[:graph.number_of_nodes("affiliation")]
    graph.nodes["field"].data["feat"] = field_emb[:graph.number_of_nodes("field")]
    graph.nodes["venue"].data["feat"] = venue_emb[:graph.number_of_nodes("venue")]

    if predict_venue:
        labels = torch.from_numpy(dataset["labels"])
    else:
        labels = torch.zeros(graph.number_of_nodes("paper"), n_classes)
        for key in dataset["labels"]:
            labels[key, dataset["labels"][key]] = 1
    train_nid, val_nid, test_nid = np.array(train_nid), np.array(val_nid), np.array(test_nid)
    return graph, labels, n_classes, train_nid, val_nid, test_nid


###############################################################################
# Code adapted from DGL HAN example:
# https://github.com/dmlc/dgl/blob/1cb210c8326fc09ac0d06edc8cee96a38ae39550/examples/pytorch/han/utils.py#L163
###############################################################################

def load_acm_raw():
    from dgl.data.utils import download, get_download_dir, _get_dgl_url
    from scipy import io as sio
    url = 'dataset/ACM.mat'
    data_path = get_download_dir() + '/ACM.mat'
    download(_get_dgl_url(url), path=data_path)

    data = sio.loadmat(data_path)
    p_vs_l = data['PvsL']       # paper-field?
    p_vs_a = data['PvsA']       # paper-author
    p_vs_t = data['PvsT']       # paper-term, bag of words
    p_vs_c = data['PvsC']       # paper-conference, labels come from that

    # We assign
    # (1) KDD papers as class 0 (data mining),
    # (2) SIGMOD and VLDB papers as class 1 (database),
    # (3) SIGCOMM and MOBICOMM papers as class 2 (communication)
    conf_ids = [0, 1, 9, 10, 13]
    label_ids = [0, 1, 2, 2, 1]

    p_vs_c_filter = p_vs_c[:, conf_ids]
    p_selected = (p_vs_c_filter.sum(1) != 0).A1.nonzero()[0]
    p_vs_l = p_vs_l[p_selected]
    p_vs_a = p_vs_a[p_selected]
    p_vs_t = p_vs_t[p_selected]
    p_vs_c = p_vs_c[p_selected]

    pa = dgl.bipartite(p_vs_a, 'paper', 'pa', 'author')
    pl = dgl.bipartite(p_vs_l, 'paper', 'pf', 'field')
    gs = [pa, pl]
    hg = dgl.hetero_from_relations(gs)

    features = torch.FloatTensor(p_vs_t.toarray())

    pc_p, pc_c = p_vs_c.nonzero()
    labels = np.zeros(len(p_selected), dtype=np.int64)
    for conf_id, label_id in zip(conf_ids, label_ids):
        labels[pc_p[pc_c == conf_id]] = label_id
    labels = torch.LongTensor(labels)

    num_classes = 3

    float_mask = np.zeros(len(pc_p))
    for conf_id in conf_ids:
        pc_c_mask = (pc_c == conf_id)
        float_mask[pc_c_mask] = np.random.permutation(np.linspace(0, 1, pc_c_mask.sum()))
    train_idx = np.where(float_mask <= 0.2)[0]
    val_idx = np.where((float_mask > 0.2) & (float_mask <= 0.3))[0]
    test_idx = np.where(float_mask > 0.3)[0]

    hg.nodes["paper"].data["feat"] = features

    return hg, labels, num_classes, train_idx, val_idx, test_idx
