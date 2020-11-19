Prepare OAG Dataset
=====================
If you would like to evaluate NARS on OAG, please follow instructions below to
prepare the dataset. The goal here is to extract graph structure and node
features and store into a more efficient format so that we can easily read and
convert to a DGLGraph object.

Dependency
----------
- dill


Download dataset
-----------------
We use the OAG dataset shared by authors of Heterogeneous Graph Transformer here:  
[https://github.com/acbull/pyHGT#oag-dataset](https://github.com/acbull/pyHGT#oag-dataset)

Please download the `graph_CS.pk` from HGT's google drive:
[https://drive.google.com/drive/folders/1a85skqsMBwnJ151QpurLFSa9o2ymc_rq](https://drive.google.com/drive/folders/1a85skqsMBwnJ151QpurLFSa9o2ymc_rq)


Extract graph and features
-----------------
ATTENTION! If your python version is >= 3.8, you might get a pickle error since
there's some change in serialization. You may need python <= 3.7.

The following steps will take a short while to finish since they all need to load
and process a 8.1 GB file.

### Extract language features for paper nodes
```python
python3 extract_paper_feats.py
```
This generates a new file `paper.npy`.

### Extract graph structure for venue prediction
```python
python3 convert_OAG_venue.py
```
This generates a new file `graph_venue.pk`.

### Extract graph structure for L1-field prediction
```python
python3 convert_OAG_field.py
```
This generates a new file `graph_field.pk`.


You might wonder why the graph structures for venue prediction and field
prediction are different.   
The reason is that in the original OAG graph shared by HGT authors, paper nodes
are connected to ground truth venue nodes and field nodes. In order to avoid
any chance of information leakage, for venue prediction task, we remove edges
between papers and venues, and for L1-field prediction tasks, we remove edges
between papers and L1-fields.
