#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

set -x
dataset=$1

if [ "$dataset" == "mag" ]; then
    embed_size=256
elif [ "$dataset" == "oag_venue" ] || [ "$dataset" == "oag_L1" ]; then
    embed_size=400
elif [ "$dataset" == "acm" ]; then
    embed_size=128
else
    echo "Unsupported dataset ${dataset}!"
    echo "Usage: bash train_graph_emb.sh [acm|mag|oag_venue|oag_L1]"
    exit -1
fi

DGLBACKEND=pytorch dglke_train \
    --model TransE_l2 \
    --batch_size 1000 \
    --neg_sample_size 200 \
    --hidden_dim $embed_size \
    --gamma 10 \
    --lr 0.1 \
    --max_step 400000 \
    --log_interval 10000 \
    -adv \
    --gpu 0 \
    --regularization_coef 1e-9 \
    --data_path ./ \
    --data_files train_triplets_$dataset \
    --format raw_udd_hrt \
    --dataset $dataset
