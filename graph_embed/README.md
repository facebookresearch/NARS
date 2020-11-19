Featurize Nodes With No Input
==========================
In academic graph datasets like ACM, MAG, and OAG, only paper nodes are
associated with input features generated from pre-trained language models. NARS
featurizes other node types with relational graph embedding.

The following commands generates TransE embedding for OGBN-MAG dataset using DGL-KE:
```bash
dataset=mag
```
Supported dataset values are `mag`, `acm`, `oag_venue`, and `oag_L1`.

Convert heterogeneous graph to triplet format (src_node_id,
edge_type, dst_node_id). This step only uses the graph structure. The node
features are not used.
```bash
python3 convert_to_triplets.py --dataset ${dataset}
```

Before generating embedding, it's better to remove existing DGL-KE
checkpoints (if any) in this folder:
```bash
rm -rf ckpts
```

The shell script `train_graph_emb.sh` uses DGL-KE to train graph embedding. The
default configuration is what we used to evaluate our paper. But feel free to
change any setting in the script like the graph embedding model, training
hyper-parameters. The training takes about 40 mins to finish on a Tesla V100 GPU.
You can also speed it up by allowing DGL-KE to use more GPUs.
```bash
bash train_graph_emb.sh ${dataset}
```

The generated embedding of all nodes will be stored in `ckpts` folder. We need
to split the graph embedding by node types and reorder back to original node
order:
```bash
python3 split_node_emb.py --dataset ${dataset}
mkdir ../TransE_${dataset}
mv *.pt ../TransE_${dataset}
```
