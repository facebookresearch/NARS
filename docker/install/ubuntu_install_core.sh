#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# install libraries for building c++ core on ubuntu
export DEBIAN_FRONTEND=noninteractive
apt update --fix-missing
apt install -y --no-install-recommends --force-yes \
        apt-utils git build-essential make cmake wget unzip sudo \
        libz-dev libxml2-dev ca-certificates vim
