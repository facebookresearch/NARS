#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# install python3
apt-get update
apt-get install -y python3 python3-dev

# install pip
cd /tmp && wget -q https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py

# santiy check
python3 --version
pip3 --version
