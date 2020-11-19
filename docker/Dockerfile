FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

COPY install/ubuntu_install_core.sh /install/ubuntu_install_core.sh
RUN bash /install/ubuntu_install_core.sh

COPY install/ubuntu_install_python.sh /install/ubuntu_install_python.sh
RUN bash /install/ubuntu_install_python.sh

COPY install/ubuntu_install_python_package.sh /install/ubuntu_install_python_package.sh
RUN bash /install/ubuntu_install_python_package.sh

# add user
RUN useradd -ms /bin/bash user

# add user to sudoers
RUN chmod u+w /etc/sudoers
RUN echo 'user\tALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
RUN chmod u-w /etc/sudoers

USER user
WORKDIR /home/user

# use python3 as default python
RUN echo 'alias python=python3' >> /home/user/.bashrc
RUN echo 'PATH=$PATH:/home/user/.local/bin' >> /home/user/.bashrc
