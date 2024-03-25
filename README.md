#  CPA - Compositional Perturbation Autoencoder 
[![PyPI version](https://badge.fury.io/py/cpa-tools.svg)](https://badge.fury.io/py/cpa-tools) [![Documentation Status](https://readthedocs.org/projects/cpa-tools/badge/?version=latest)](https://cpa-tools.readthedocs.io/en/latest/?badge=latest) [![Downloads](https://static.pepy.tech/badge/cpa-tools)](https://pepy.tech/project/cpa-tools)

## What is CPA?

![Alt text](https://user-images.githubusercontent.com/33202701/156530222-c61e5982-d063-461c-b66e-c4591d2d0de4.png?raw=true "Title")

`CPA` is a framework to learn the effects of perturbations at the single-cell level. CPA encodes and learns phenotypic drug responses across different cell types, doses, and combinations. CPA allows:

* Out-of-distribution predictions of unseen drug and gene combinations at various doses and among different cell types.
* Learn interpretable drug and cell-type latent spaces.
* Estimate the dose-response curve for each perturbation and their combinations.
* Transfer pertubration effects from on cell-type to an unseen cell-type.
* Enable batch effect removal on a latent space and also gene expression space.


## Installation



### Installing CPA
You can install CPA using pip and also directly from the github to access latest development version. 
See detailed instructions [here](https://cpa-tools.readthedocs.io/en/latest/installation.html). 

## How to use CPA
Several tutorials are available [here](https://cpa-tools.readthedocs.io/en/latest/tutorials/index.html) to get you started with CPA.
The following table contains the list of tutorials:

|Description | Link |
| --- | --- |
| Predicting combinatorial drug perturbations | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/theislab/cpa/blob/master/docs/tutorials/combosciplex.ipynb) - [![Open In Documentation](https://img.shields.io/badge/docs-blue)](https://cpa-tools.readthedocs.io/en/latest/tutorials/combosciplex.html) |
| Predicting unseen perturbations uisng external embeddings enabling the model to predict unseen reponses to unseen drugs| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/theislab/cpa/blob/master/docs/tutorials/combosciplex_Rdkit_embeddings.ipynb) - [![Open In Documentation](https://img.shields.io/badge/docs-blue)](https://cpa-tools.readthedocs.io/en/latest/tutorials/combosciplex_Rdkit_embeddings.html) |
|Predicting combinatorial CRISPR perturbations| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/theislab/cpa/blob/master/docs/tutorials/Norman.ipynb) - [![Open In Documentation](https://img.shields.io/badge/docs-blue)](https://cpa-tools.readthedocs.io/en/latest/tutorials/Norman.html) |
|Context transfer (i.e. predict the effect of a perturbation (e.g. disease) on unseen cell types or transfer perturbation effects from one context to another) demo on IFN-β scRNA perturbation dataset | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/theislab/cpa/blob/master/docs/tutorials/Kang.ipynb) - [![Open In Documentation](https://img.shields.io/badge/docs-blue)](https://cpa-tools.readthedocs.io/en/latest/tutorials/Kang.html) |
|Batch effect removal in gene expression and latent space| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/theislab/cpa/blob/master/docs/tutorials/Batch_correction_in_expression_space.ipynb) - [![Open In Documentation](https://img.shields.io/badge/docs-blue)](https://cpa-tools.readthedocs.io/en/latest/tutorials/Batch_correction_in_expression_space.html) |

How to optmize CPA hyperparamters for your data
-----------------------------------------------
We provide an example script to use the built-in hyperparameter optimization function in CPA (based on scvi-tools hyperparam optimizer). You can find the script at `examples/tune_script.py`.

After the hyperparameter optimization using tune_script.py is done, `result_grid.pkl` is saved in your current directory using the `pickle` library. You can load the results using the following code:

```python
import pickle
with open('result_grid.pkl', 'rb') as f:
    result_grid = pickle.load(f)
```
From here, you can follow the instructions in the [Ray Documentations](https://docs.ray.io/en/latest/tune/examples/tune_analyze_results.html#experiment-level-analysis-working-with-resultgrid) to analyze the run, and choose the best hyperparameters for your data.

You can also use the integration with wandb to log the hyperparameter optimization results. You can find the script at `examples/tune_script_wandb.py`. --> `use_wandb=True`

Everything is based on [Ray Tune](https://ray.io/). You can find more information about the hyperparameter optimization in the [Ray Tune Documentations](https://docs.ray.io/en/latest/tune/index.html).

The tuner is adapted and adjusted from scvi-tools v1.2.0 (unreleased) [release notes](https://docs.scvi-tools.org/en/stable/release_notes/index.html)


Datasets and Pre-trained models
-------------------------------
Datasets and pre-trained models are available [here](https://drive.google.com/drive/folders/1yFB0gBr72_KLLp1asojxTgTqgz6cwpju?usp=drive_link).


Recepie for Pre-processing a custom scRNAseq perturbation dataset
-----------------------------------------------------------------
If you have access to you raw data, you can do the following steps to pre-process your dataset. A raw dataset should be a [scanpy](https://scanpy.readthedocs.io/en/stable/) object containing raw counts and available required metadata (i.e. perturbation, dosage, etc.).

Pre-processing steps
--------------------
0. Check for required information in cell metadata:
    a) Perturbation information should be in `adata.obs`.
    b) Dosage information should be in `adata.obs`. In cases like CRISPR gene knockouts, disease states, time perturbations, etc, you can create & add a dummy dosage in your `adata.obs`. For example:
    ```python
        adata.obs['dosage'] = adata.obs['perturbation'].astype(str).apply(lambda x: '+'.join(['1.0' for _ in x.split('+')])).values
    ```
    c) [If available] Cell type information should be in `adata.obs`.
    d) [**Multi-batch** integration] Batch information should be in `adata.obs`.

1. Filter out cells with low number of counts (`sc.pp.filter_cells`). For example:
    ```python
    sc.pp.filter_cells(adata, min_counts=100)
    ```

    [optional]
    ```python
    sc.pp.filter_genes(adata, min_counts=5)
    ```
    
2. Save the raw counts in `adata.layers['counts']`.
    ```python
    adata.layers['counts'] = adata.X.copy()
    ```
3. Normalize the counts (`sc.pp.normalize_total`).
    ```python
    sc.pp.normalize_total(adata, target_sum=1e4, exclude_highly_expressed=True)
    ```
4. Log transform the normalized counts (`sc.pp.log1p`).
    ```python
    sc.pp.log1p(adata)
    ```
5. Highly variable genes selection:
    There are two options:
        1. Use the `sc.pp.highly_variable_genes` function to select highly variable genes.
        ```python
            sc.pp.highly_variable_genes(adata, n_top_genes=5000, subset=True)
        ```
        2. (**Highly Recommended** specially for **Multi-batch** integration scenarios) Use scIB's [highly variable genes selection](https://scib.readthedocs.io/en/latest/api/scib.preprocessing.hvg_batch.html#scib.preprocessing.hvg_batch) function to select highly variable genes. This function is more robust to batch effects and can be used to select highly variable genes across multiple datasets.
        ```python
            import scIB
            adata_hvg = scIB.pp.hvg_batch(adata, batch_key='batch', n_top_genes=5000, copy=True)
        ```


Congrats! Now you're dataset is ready to be used with CPA. Don't forget to save your pre-processed dataset using `adata.write_h5ad` function.


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

