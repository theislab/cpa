# Causal CPA

## Code structure

- `cpa/_module.py` contains the core architecture of the CPA model, as well as components to compute vital quantities, including reconstruction errors/ELBOs, basal latent representations, drug embeddings, and gene expression. It also contains adversarial classifiers.
- `cpa/_task.py` describes how to train the model parameters. 
- `cpa/_utils.py` includes several useful classes (architectures) and functions to prepare the AnnData to be used for training.
- `cpa/_model.py` is the user interface to instantiate and train the model, as well as for inference.

## Notebooks


## Pierre's implementation vs. Facebook's CPA
Here are the most straightforward differences existing between the two implementations.

- While not necessarily different, I am not sure that the fully connected brick component is implemented in the same way as in codebase.
- A single optimizer has used for for the dosers and (V)AE.
- The decoder's means and standard deviations may not be computed in the exact same way as in the reference codebase.
- ? While properly running, this codebase does not have the same behavior as the reference yet.
- The treatments are contained in a matrix instead of a drug/treatment representation.
- The doser MLP is not featured.
- The NB distribution should not be trusted yet, because the data is often normalized, and hence not count-based.


## To-Do list

- [x] create code structure like scvi-tools-skeleton
- [ ] Check `FCLayers` implementation vs. CPA's MLP class
- [x] Add another optimizer for dosers network
- [ ] Check decoder's output (mean and variance) vs. Normal Distribution
- [ ] Check Why the scVI's wrapper version of CPA is much more time expensive and memory consuming.
- [x] Pre-process dataset and add drug-dose matrix in `obsm` in scVI's wrapper
- [x] Implement MLP for dosers
- [ ] Check NB distribution
- [x] `log(1 + exp(variance)) + epsilon` is not implemented in Facebook implementation of CPA
    - Why this will not result in negative input for `variance.log()` in `GaussLoss`?
- [x] Multiple covariate handling in a clean manner in module, dataset, and trainer.
- [ ] Train network with `NB` loss
- [ ] Reproducibility
    - [ ] Norman 2019
    - [ ] Pachter
    - [ ] Trapnell


