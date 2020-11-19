Sample Relation Subsets
=====================
NARS generates multi-hop neighbor-averaged features on `K` randomly sampled
subsets. The python script in this folder performs the sampling.

For example, the following command samples 8 relation subsets for OGBN-MAG dataset:
```bash
python3 sample_random_subsets.py --num-subsets 8 --dataset mag
```

The supported datasets are `acm`, `mag`, `oag_L1`, and `oag_venue`. For OAG
dataset, make sure you have preprocessed the dataset following instructions in
[oag_dataset](../oag_dataset) folder.

[examples](./examples) folder contains the relation subsets that we previously
sampled.
