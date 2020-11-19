# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import time
import torch
import torch.nn as nn
import numpy as np
import logging
from data import load_data, read_relation_subsets, gen_rel_subset_feature
from model import SIGN, PartialWeightedAggregator
from utils import get_n_params, get_evaluator, train, test


def preprocess_agg(g, metapaths, args, device, aggregator):
    num_paper, feat_size = g.nodes["paper"].data["feat"].shape
    new_feats = [torch.zeros(num_paper, feat_size) for _ in range(args.R + 1)]
    print("Start generating features for each sub-metagraph:")
    for path_id, mpath in enumerate(metapaths):
        print(mpath)
        feats = gen_rel_subset_feature(g, mpath, args, device)
        for i in range(args.R + 1):
            feat = feats[i]
            feat *= aggregator.weight_store[i][path_id].unsqueeze(0)
            new_feats[i] += feat
        feats = None
    return new_feats


def recompute_selected_subsets(g, selected_subsets, args, num_nodes, feat_size, device):
    # TODO: recompute in parallel using multi-processing
    # Or we should save neighbor-averaged features to disk and load them back instead of re-computing
    start = time.time()
    with torch.no_grad():
        feats = [
            torch.zeros(num_nodes, len(selected_subsets), feat_size)
            for _ in range(args.R + 1)
        ]
        for i, subset in enumerate(selected_subsets):
            rel_feats = gen_rel_subset_feature(g, subset, args, device)
            for hop in range(args.R + 1):
                feats[hop][:, i, :] = rel_feats[i]
    end = time.time()
    print("Recompute takes {:.4f} sec".format(end - start))
    return feats


def main(args):
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    if args.gpu < 0:
        device = "cpu"
    else:
        device = f"cuda:{args.gpu}"

    # Load dataset
    data = load_data(device, args)
    g, labels, num_classes, train_nid, val_nid, test_nid = data
    evaluator = get_evaluator(args.dataset)

    rel_subsets = read_relation_subsets(args.use_relation_subsets)
    num_feats = len(rel_subsets)
    in_feats = g.nodes["paper"].data["feat"].shape[1]
    num_paper = g.number_of_nodes("paper")
    num_hops = args.R + 1  # include self feature hop 0

    aggregator = PartialWeightedAggregator(
        num_feats, in_feats, num_hops, args.sample_size
    )

    # Preprocess neighbor-averaged features over sampled relation subgraphs
    with torch.no_grad():
        history_sum = preprocess_agg(g, rel_subsets, args, device, aggregator)
        print("Done preprocessing")
    labels = labels.to(device)

    # Set up logging
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
    logging.info(str(args))

    # Create model
    model = nn.Sequential(
        aggregator,
        SIGN(
            in_feats,
            args.num_hidden,
            num_classes,
            num_hops,
            args.ff_layer,
            args.dropout,
            args.input_dropout,
        ),
    )
    logging.info("# Params: {}".format(get_n_params(model)))
    model.to(device)

    if len(labels.shape) == 1:
        # single label multi-class
        loss_fcn = nn.NLLLoss()
    else:
        # multi-label multi-class
        loss_fcn = nn.KLDivLoss(reduction="batchmean")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    with torch.no_grad():
        selected = np.random.choice(num_feats, args.sample_size, replace=False)
        selected_subsets = [rel_subsets[i] for i in selected]
        feats_selected = recompute_selected_subsets(
            g, selected_subsets, args, num_paper, in_feats, device
        )

    # Start training
    best_epoch = 0
    best_val = 0
    for epoch in range(1, args.num_epochs + 1):
        start = time.time()
        model.train()
        train(
            model,
            feats_selected,
            labels,
            train_nid,
            loss_fcn,
            optimizer,
            args.batch_size,
            history=history_sum,
        )

        if epoch % args.eval_every == 0:
            with torch.no_grad():
                train_res, val_res, test_res = test(
                    model,
                    feats_selected,
                    labels,
                    train_nid,
                    val_nid,
                    test_nid,
                    evaluator,
                    args.eval_batch_size,
                    history=history_sum,
                )
            end = time.time()
            val_acc = val_res[0]
            log = "Epoch {}, Times(s): {:.4f}".format(epoch, end - start)
            if args.dataset.startswith("oag"):
                log += ", NDCG: Train {:.4f}, Val {:.4f}, Test {:.4f}".format(
                    train_res[0], val_res[0], test_res[0]
                )
                log += ", MRR: Train {:.4f}, Val {:.4f}, Test {:.4f}".format(
                    train_res[1], val_res[1], test_res[1]
                )
            else:
                log += ", Accuracy: Train {:.4f}, Val {:.4f}, Test {:.4f}".format(
                    train_res[0], val_res[0], test_res[0]
                )
            logging.info(log)
            if val_acc > best_val:
                best_val = val_acc
                best_epoch = epoch

        # update history and aggregation weight and resample
        if epoch % args.resample_every == 0:
            with torch.no_grad():
                aggregator.cpu()
                history_sum = aggregator((feats_selected, history_sum))
                aggregator.update_selected(selected)
                aggregator.to(device)
                selected = np.random.choice(num_feats, args.sample_size, replace=False)
                selected_subsets = [rel_subsets[i] for i in selected]
                feats_selected = recompute_selected_subsets(
                    g, selected_subsets, args, num_paper, in_feats, device
                )

    logging.info("Best Epoch {}, Val {:.4f}".format(best_epoch, best_val))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Neighbor-Averaging over Relation Subgraphs"
    )
    parser.add_argument("--num-epochs", type=int, default=1000)
    parser.add_argument("--num-hidden", type=int, default=256)
    parser.add_argument("--R", type=int, default=2, help="number of hops")
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--dataset", type=str, default="oag")
    parser.add_argument(
        "--data-dir", type=str, default=None, help="path to dataset, only used for OAG"
    )
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=50000)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=50000,
        help="evaluation batch size, -1 for full batch",
    )
    parser.add_argument(
        "--ff-layer", type=int, default=2, help="number of feed-forward layers"
    )
    parser.add_argument("--input-dropout", action="store_true")
    parser.add_argument("--use-emb", required=True, type=str)
    parser.add_argument("--use-relation-subsets", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=3)
    parser.add_argument("--resample-every", type=int, default=10)
    parser.add_argument(
        "--cpu-preprocess", action="store_true", help="Preprocess on CPU"
    )
    args = parser.parse_args()

    print(args)
    main(args)
