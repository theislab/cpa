# CPA - Compositional Perturbation Autoencoder


## What is CPA?

![Alt text](https://user-images.githubusercontent.com/33202701/156530222-c61e5982-d063-461c-b66e-c4591d2d0de4.png?raw=true "Title")

`CPA` is a framework to learn effects of perturbations at the single-cell level. CPA encodes and learns phenotypic drug response across different cell types, doses and drug combinations. CPA allows:

* Out-of-distribution predictions of unseen drug combinations at various doses and among different cell types.
* Learn interpretable drug and cell type latent spaces.
* Estimate dose response curve for each perturbation and their combinations.
* Access the uncertainty of the estimations of the model.


Usage and installation
-------------------------------
See [here](https://cpa-tools.readthedocs.io/en/latest/index.html) for documentation and tutorials.

Support and contribute
-------------------------------
If you have a question or new architecture or a model that could be integrated into our pipeline, you can
post an [issue](https://github.com/theislab/cpa/issues/new)


###### Acknowledgment
This code is inspired by an early implementatiom by [Pierre Boyeau](https://github.com/PierreBoyeau) using [scvi-tools](https://scvi-tools.org/).

