#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# install libraries for python package on ubuntu
pip3 install numpy cython scipy networkx matplotlib nltk pandas rdflib requests[security] dill

# install DL Framework
pip3 install torch==1.5.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# install DGL
pip3 install dgl-cu101==0.4.3.post2

# install ogb
pip3 install ogb==1.2.1

# install dgl-ke
pip3 install dglke==0.1.0
