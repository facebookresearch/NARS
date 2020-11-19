# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


###############################################################################
# Training and testing for one epoch
###############################################################################

def train(model, feats, labels, train_nid, loss_fcn, optimizer, batch_size, history=None):
    model.train()
    device = labels.device
    dataloader = torch.utils.data.DataLoader(
        train_nid, batch_size=batch_size, shuffle=True, drop_last=False)
    for batch in dataloader:
        batch_feats = [x[batch].to(device) for x in feats]
        if history is not None:
            # Train aggregator partially using history
            batch_feats = (batch_feats, [x[batch].to(device) for x in history])
        loss = loss_fcn(model(batch_feats), labels[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(model, feats, labels, train_nid, val_nid, test_nid, evaluator, batch_size, history=None):
    model.eval()
    num_nodes = labels.shape[0]
    device = labels.device
    dataloader = torch.utils.data.DataLoader(
        torch.arange(num_nodes), batch_size=batch_size,
        shuffle=False, drop_last=False)
    scores = []
    for batch in dataloader:
        batch_feats = [feat[batch].to(device) for feat in feats]
        if history is not None:
            # Train aggregator partially using history
            batch_feats = (batch_feats, [x[batch].to(device) for x in history])
        pred = model(batch_feats)
        scores.append(evaluator(pred, labels[batch]))
    # For each evaluation metric, concat along node dimension
    metrics = [torch.cat(s, dim=0) for s in zip(*scores)]
    train_res = compute_mean(metrics, train_nid)
    val_res = compute_mean(metrics, val_nid)
    test_res = compute_mean(metrics, test_nid)
    return train_res, val_res, test_res


###############################################################################
# Evaluator for different datasets
###############################################################################

def batched_acc(pred, labels):
    # testing accuracy for single label multi-class prediction
    return (torch.argmax(pred, dim=1) == labels,)


def get_evaluator(dataset):
    dataset = dataset.lower()
    if dataset.startswith("oag"):
        return batched_ndcg_mrr
    else:
        return batched_acc


def compute_mean(metrics, nid):
    num_nodes = len(nid)
    return [m[nid].float().sum().item() / num_nodes for  m in metrics]


###############################################################################
# Original implementation of evaluation metrics NDCG and MRR by HGT author
# https://github.com/acbull/pyHGT/blob/f7c4be620242d8c1ab3055f918d4c082f5060e07/OAG/pyHGT/utils.py
###############################################################################

def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
    return 0.


def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max


def mean_reciprocal_rank(rs):
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])


def ndcg_mrr(pred, labels):
    """
    Compute both NDCG and MRR for single-label and multi-label. Code extracted from
    https://github.com/acbull/pyHGT/blob/f7c4be620242d8c1ab3055f918d4c082f5060e07/OAG/train_paper_venue.py#L316 (single label)
    and
    https://github.com/acbull/pyHGT/blob/f7c4be620242d8c1ab3055f918d4c082f5060e07/OAG/train_paper_field.py#L322 (multi-label)
    """
    test_res = []
    if len(labels.shape) == 1:
        # single-label
        for ai, bi in zip(labels, pred.argsort(descending = True)):
            test_res += [(bi == ai).int().tolist()]
    else:
        # multi-label
        for ai, bi in zip(labels, pred.argsort(descending = True)):
            test_res += [ai[bi].int().tolist()]
    ndcg = np.mean([ndcg_at_k(resi, len(resi)) for resi in test_res])
    mrr = mean_reciprocal_rank(test_res)
    return ndcg, mrr


###############################################################################
# Fast re-implementation of NDCG and MRR for a batch of nodes.
# We provide unit test below using random input to verify correctness /
# equivalence.
###############################################################################

def batched_dcg_at_k(r, k):
    assert(len(r.shape) == 2 and r.size != 0 and k > 0)
    r = r[:, :k].float()
    # Usually, one defines DCG = \sum\limits_{i=0}^{n-1}\frac{r_i}/{log2(i+2)}
    # So, we should
    # return (r / torch.log2(torch.arange(0, r.shape[1], device=r.device, dtype=r.dtype).view(1, -1) + 2)).sum(dim=1)
    # However, HGT author implements DCG = r_0 + \sum\limits_{i=1}^{n-1}\frac{r_i}/{log2(i+1)}, which makes DCG and NDCG larger
    # Here, we follow HGT author for a fair comparison
    return r[:, 0] + (r[:, 1:] / torch.log2(torch.arange(1, r.shape[1], device=r.device, dtype=r.dtype).view(1, -1) + 1)).sum(dim=1)


def batched_ndcg_at_k(r, k):
    dcg_max = batched_dcg_at_k(r.sort(dim=1, descending=True)[0], k)
    dcg_max_inv = 1.0 / dcg_max
    dcg_max_inv[torch.isinf(dcg_max_inv)] = 0
    return batched_dcg_at_k(r, k) * dcg_max_inv


def batched_mrr(r):
    r = r != 0
    # torch 1.5 does not guarantee max returns first occurrence
    # https://pytorch.org/docs/1.5.0/torch.html?highlight=max#torch.max
    # So we get first occurrence of non-zero using numpy max
    max_indices = torch.from_numpy(r.cpu().numpy().argmax(axis=1))
    max_values = r[torch.arange(r.shape[0]), max_indices]
    r = 1.0 / (max_indices.float() + 1)
    r[max_values == 0] = 0
    return r


def batched_ndcg_mrr(pred, labels):
    pred = pred.argsort(descending=True)
    if len(labels.shape) == 1:
        # single-label
        labels = labels.view(-1, 1)
        rel = (pred == labels).int()
    else:
        # multi-label
        rel = torch.gather(labels, 1, pred)
    return batched_ndcg_at_k(rel, rel.shape[1]), batched_mrr(rel)


###############################################################################
# Unit test for our re-implementation of NDCG and MRR using random input
###############################################################################

def isclose(a, b, rtol=1e-4, atol=1e-4):
    print(a, b)
    return abs(a - b) <= atol + rtol * abs(b)


def test_single_label(num_nodes, num_classes, device):
    labels = torch.from_numpy(np.random.choice(num_classes, num_nodes))
    pred = torch.randn(num_nodes, num_classes)
    ndcg1, mrr1 = ndcg_mrr(pred, labels)
    pred = pred.to(device)
    labels = labels.to(device)
    ndcg2, mrr2 = batched_ndcg_mrr(pred, labels)
    assert isclose(ndcg1, ndcg2.mean().item()) and isclose(mrr1, mrr2.mean().item())


def test_multi_label(num_nodes, num_classes, device):
    # Generate multi-label with 20% 1s and 80% 0s
    labels = torch.from_numpy(np.random.choice([0, 1], size=(num_nodes, num_classes), p=[0.8, 0.2]))
    pred = torch.randn(num_nodes, num_classes)
    ndcg1, mrr1 = ndcg_mrr(pred, labels)
    pred = pred.to(device)
    labels = labels.to(device)
    ndcg2, mrr2 = batched_ndcg_mrr(pred, labels)
    assert isclose(ndcg1, ndcg2.mean().item()) and isclose(mrr1, mrr2.mean().item())


if __name__ == "__main__":
    """
    Main module for checking correctness of NDCG and MRR
    """

    import argparse
    import time

    # take current milli-second as random seed
    seed = int(time.time() * 1e3) % 1000
    np.random.seed(seed)
    torch.manual_seed(seed)

    parser = argparse.ArgumentParser("Unit test of NDCG and MRR")
    parser.add_argument("--num-classes", type=int, default=1000,
                        help="Number of classes")
    parser.add_argument("--num-nodes", type=int, default=5000,
                        help="Number of nodes")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device ID. -1 for CPU")
    args = parser.parse_args()

    if args.gpu < 0:
        device = "cpu"
    else:
        device = f"cuda:{args.gpu}"

    test_single_label(args.num_nodes, args.num_classes, device)
    test_multi_label(args.num_nodes, args.num_classes, device)

