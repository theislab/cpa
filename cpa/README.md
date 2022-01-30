# CPA model

## Code structure

- `_module.py` contains the core architecture of the CPA model, as well as components to compute vital quantities, including reconstruction errors/ELBOs, basal latent representations, drug embeddings, and gene expression.
- `_task.py` describes how to train the model parameters. It also contains adversarial classifiers.
- `_utils.py` includes several useful classes (architectures) and functions to prepare the AnnData to be used for training.
- `_model.py` is the user interface to instantiate and train the model, as well as for inference.

## Notebooks

Very minimal now. One single and one combo notebooks can be found in this folder.

    
## Spotted differences between implementations
Here are the most straightforward differences existing between the two implementations.

- While not necessarily different, I am not sure that the fully connected brick component is implemented in the same way as in 
- I used a single optimizer for the dosers and (V)AE.
- The decoder's means and standard deviations may not be computed in the exact same way as in the reference codebase.
- ? While properly running, this codebase does not have the same behavior as the reference yet.
- The treatments are contained in a matrix instead of a drug/treatment representation.
- The doser `MLP` is not featured.
- The NB distribution should not be trusted yet, because the data is often normalized, and hence not count-based.