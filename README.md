#  CPA - Compositional Perturbation Autoencoder [![PyPI version](https://badge.fury.io/py/cpa-tools.svg)](https://badge.fury.io/py/cpa-tools) [![Documentation Status](https://readthedocs.org/projects/cpa-tools/badge/?version=latest)](https://cpa-tools.readthedocs.io/en/latest/?badge=latest) [![Downloads](https://static.pepy.tech/badge/cpa-tools)](https://pepy.tech/project/cpa-tools)

## What is CPA?

![Alt text](https://user-images.githubusercontent.com/33202701/156530222-c61e5982-d063-461c-b66e-c4591d2d0de4.png?raw=true "Title")

`CPA` is a framework to learn the effects of perturbations at the single-cell level. CPA encodes and learns phenotypic drug responses across different cell types, doses, and combinations. CPA allows:

* Out-of-distribution predictions of unseen drug and gene combinations at various doses and among different cell types.
* Learn interpretable drug and cell-type latent spaces.
* Estimate the dose-response curve for each perturbation and their combinations.
* Transfer pertubration effects from on cell-type to an unseen cell-type.


## Installation

### Requirements 

#### Conda Environment
We recommend using [Anaconda](https://www.anaconda.com/)/[Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) to create a conda environment for using CPA. You can create a python environment using the following command:

```bash
conda create -n cpa python=3.8
```

Then, you can activate the environment using:

```bash
conda activate cpa
```
#### Pytorch
CPA is implemented in Pytorch and **requires Pytorch version >= 1.13.1**.

##### OSX
```bash
pip install torch==1.13.1
```
##### Linux and Windows
If you have access to GPUs, you can install the GPU version of Pytorch following the instructions [here](https://pytorch.org/get-started/previous-versions/).

Sample command for installing Pytorch 1.13.1 on different CUDA versions:

```bash
# ROCM 5.2 (Linux only)
pip install torch==1.13.1+rocm5.2 --extra-index-url https://download.pytorch.org/whl/rocm5.2
# CUDA 11.6
pip install torch==1.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
# CUDA 11.7
pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
# CPU only
pip install torch==1.13.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu
```

### Installing CPA
You can install CPA using pip:

```bash
pip install cpa-tools
```


## How to use CPA
Several tutorials are available [here](https://cpa-tools.readthedocs.io/en/latest/tutorials/index.html) to get you started with CPA.
The following table contains the list of tutorials:

| Dataset | Year | Description | Link |
| --- | --- | --- | --- |
| Combo Sci-Plex | 2022 | CPA Training demo on combo sciplex | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/theislab/cpa/blob/master/docs/tutorials/combosciplex.ipynb) - [![Open In Documentation](https://img.shields.io/badge/docs-blue)](https://cpa-tools.readthedocs.io/en/latest/tutorials/combosciplex.html) |
| Combo Sci-Plex | 2022 | CPA Training demo on combo sciplex with rdkit embeddings as pretrained drug embeddings | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/theislab/cpa/blob/master/docs/tutorials/combosciplex_Rdkit_embeddings.ipynb) - [![Open In Documentation](https://img.shields.io/badge/docs-blue)](https://cpa-tools.readthedocs.io/en/latest/tutorials/combosciplex_Rdkit_embeddings.html) |
| Norman et al. | 2019 | CPA Training demo on Norman CRISPR scRNA perturbation dataset | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/theislab/cpa/blob/master/docs/tutorials/Norman.ipynb) - [![Open In Documentation](https://img.shields.io/badge/docs-blue)](https://cpa-tools.readthedocs.io/en/latest/tutorials/Norman.html) |
| Kang et al. | 2018 | Context transfer (i.e. predict the effect of a perturbation (e.g. disease) on unseen cell types or transfer perturbation effects from one context to another) demo on IFN-β scRNA perturbation dataset | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/theislab/cpa/blob/master/docs/tutorials/Kang.ipynb) - [![Open In Documentation](https://img.shields.io/badge/docs-blue)](https://cpa-tools.readthedocs.io/en/latest/tutorials/Kang.html) |

How to optmize CPA hyperparamters for your data
-------------------------------


Datasets and Pre-trained models
-------------------------------
Datasets and pre-trained models are available [here](https://drive.google.com/drive/folders/1yFB0gBr72_KLLp1asojxTgTqgz6cwpju?usp=drive_link).


Support and contribute
-------------------------------
If you have a question or new architecture or a model that could be integrated into our pipeline, you can
post an [issue](https://github.com/theislab/cpa/issues/new)

Reference
-------------------------------
If CPA is helpful in your research, please consider citing the  [Lotfollahi et al. 2023](https://www.embopress.org/doi/full/10.15252/msb.202211517)


    @article{lotfollahi2023predicting,
        title={Predicting cellular responses to complex perturbations in high-throughput screens},
        author={Lotfollahi, Mohammad and Klimovskaia Susmelj, Anna and De Donno, Carlo and Hetzel, Leon and Ji, Yuge and Ibarra, Ignacio L and Srivatsan, Sanjay R and Naghipourfar, Mohsen and Daza, Riza M and 
        Martin, Beth and others},
        journal={Molecular Systems Biology},
        pages={e11517},
        year={2023}
    }

