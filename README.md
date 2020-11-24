Neighbor Averaging over Relation Subgraphs (NARS)
=================
NARS is an algorithm for node classification on heterogeneous graphs, based on
scalable neighbor averaging techniques that have been previously used in e.g.
[SIGN](https://arxiv.org/abs/2004.11198) to heterogeneous scenarios by
generating neighbor-averaged features on sampled relation induced subgraphs.

For more details, please check out our paper:

[Scalable Graph Neural Networks for Heterogeneous Graphs](https://arxiv.org/abs/2011.09679)


Setup
-------------------
### Dependencies
- torch==1.5.1+cu101
- dgl-cu101==0.4.3.post2
- ogb==1.2.1
- dglke==0.1.0


### Docker
We have prepared a dockerfile for building a container with clean environment
and all required dependencies. Please checkout instructions in
[docker](./docker).


Data Preparation
------------------------
### Download and pre-process OAG dataset (optional)
If you plan to evaluate on OAG dataset, you need to follow instructions in
[oag_dataset](./oag_dataset) to download and pre-process dataset.

### Generate input for featureless node types
In academic graph datasets (ACM, MAG, OAG) in which only paper nodes are
associated with input features. NARS featurizes other node types with TransE
relational graph embedding pre-trained on the graph structure.

Please follow instructions in [graph_embed](./graph_embed) to generate
embeddings for each dataset.

### Sample relation subsets
NARS samples Relation Subsets (see our paper for details). Please follow the
instructions in [sample_relation_subsets](./sample_relation_subsets) to
generate these subsets.

Or you may skip this step and use the [example
subsets](./sample_relation_subsets/examples) that have added to this
repository.

Run NARS Experiments
------------------------
NARS are evaluated on three academic graph datasets to predict publishing
venues and fields of papers.

### ACM
```bash
python3 train.py --dataset acm --use-emb TransE_acm --R 2 \
    --use-relation-subsets sample_relation_subsets/examples/acm \
    --num-hidden 64 --lr 0.003 --dropout 0.7 --eval-every 1 \
    --num-epochs 100 --input-dropout
```

### OGBN-MAG
```bash
python3 train.py --dataset mag --use-emb TransE_mag --R 5 \
    --use-relation-subset sample_relation_subsets/examples/mag \
    --eval-batch-size 50000 --num-hidden 512 --lr 0.001 --batch-s 50000 \
    --dropout 0.5 --num-epochs 1000
```

### OAG (venue prediction)
```bash
python3 train.py --dataset oag_venue --use-emb TransE_oag_venue --R 3 \
    --use-relation-subsets sample_relation_subsets/examples/oag_venue \
    --eval-batch-size 25000 --num-hidden 256 --lr 0.001 --batch-size 1000 \
    --data-dir oag_dataset --dropout 0.5 --num-epochs 200
```

### OAG (L1-field prediction)
```bash
python3 train.py --dataset oag_L1 --use-emb TransE_oag_L1 --R 3 \
    --use-relation-subsets sample_relation_subsets/examples/oag_L1 \
    --eval-batch-size 25000 --num-hidden 256 --lr 0.001 --batch-size 1000 \
    --data-dir oag_dataset --dropout 0.5 --num-epochs 200
```

### Results
Here is a summary of model performance using [example relation
subsets](./sample_relation_subsets/examples):

For ACM and OGBN-MAG dataset, the task is to predict paper publishing venue.
| Dataset    | # Params   | Test Accuracy   |
| :--------: | :--------: | :-------------: |
| ACM        | 0.40M      | 0.9305±0.0043   |
| OGBN-MAG   | 4.13M      | 0.5240±0.0016   |

For OAG dataset, there are two different node predictions tasks: predicting
venue (single-label) and L1-field (multi-label). And we follow Heterogeneous
Graph Transformer to evaluate using NDCG and MRR metrics.
| Task       | # Params   | NDCG            | MRR             |
| :--------: | :--------: | :-------------: | :-------------: |
| Venue      | 2.24M      | 0.5214±0.0010   | 0.3434±0.0012   |
| L1-field   | 1.41M      | 0.86420.0022    | 0.8542±0.0019   |



### Run with limited GPU memory
The above commands were tested on Tesla V100 (32 GB) and Tesla T4 (15GB). If
your GPU memory isn't enough for handling large graphs, try the following:
- add `--cpu-process` to the command to move preprocessing logic to CPU
- reduce evaluation batch size with `--eval-batch-size`. The evaluation result won't be affected since model is fixed.
- reduce training batch with `--batch-size`


Run NARS with Reduced CPU Memory Footprint
------------------------
As mentioned in our paper, using a lot of relation subsets may consume too much
CPU memory. To reduce CPU memory footprint, we implemented an optimization in
`train_partial.py` which trains part of our feature aggregation weights at a
time.

Using OGBN-MAG dataset as an example, the following command randomly picks 3
subsets from all 8 sampled relation subsets and trains their aggregation
weights every 10 epochs.
```bash
python3 train_partial.py --dataset mag --use-emb TransE_mag --R 5 \
    --use-relation-subsets sample_relation_subsets/examples/mag \
    --eval-batch-size 50000 --num-hidden 512 --lr 0.001 --batch-size 50000 \
    --dropout 0.5 --num-epochs 1000 --sample-size 3 --resample-every 10
```

Citation
--------------------------
Please cite our paper with:
```tex
@article{yu2020scalable,
    title={Scalable Graph Neural Networks for Heterogeneous Graphs},
    author={Yu, Lingfan and Shen, Jiajun and Li, Jinyang and Lerer, Adam},
    journal={arXiv preprint arXiv:2011.09679},
    year={2020}
}
```

License
--------------------------
NARS is CC-by-NC licensed, as found in the LICENSE file.
