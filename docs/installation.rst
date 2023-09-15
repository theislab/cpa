Installation
============

Prerequisites
~~~~~~~~~~~~~~

Conda Environment
#################
We recommend using `Anaconda <https://www.anaconda.com/>`_ / `Miniconda <https://docs.conda.io/projects/miniconda/en/latest/>`_ to create a conda environment for using CPA. You can create a python environment using the following command:

    conda create -n cpa python=3.9

Then, you can activate the environment using:

    conda activate cpa

Pytorch
########
CPA is implemented in Pytorch and **requires Pytorch version >= 1.13.1**.

OSX
---
You can install Pytorch 1.13.1 using the following command:

    pip install torch==1.13.1

Linux and Windows
-----------------

If you have access to GPUs, you can install the GPU version of Pytorch following the instructions `here <https://pytorch.org/get-started/previous-versions/>`_ .

Sample command for installing Pytorch 1.13.1 on different CUDA versions:

    # ROCM 5.2 (Linux only)

    pip install torch==1.13.1+rocm5.2 --extra-index-url https://download.pytorch.org/whl/rocm5.2

    # CUDA 11.6
    
    pip install torch==1.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
    
    # CUDA 11.7
    
    pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
    
    # CPU only
    
    pip install torch==1.13.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu


Installing CPA
##############
Finally, You can install latest version of CPA using pip:

    pip install cpa-tools --upgrade
