CPA - Compositional Perturbation Autoencoder 
#######################################################

.. image:: https://user-images.githubusercontent.com/33202701/156530222-c61e5982-d063-461c-b66e-c4591d2d0de4.png?raw=true
  :width: 800
  :alt: The architecture of CPA

What is CPA?
************

`CPA` is a framework to learn effects of perturbations at the single-cell level. CPA encodes and learns phenotypic drug response across different cell types, doses and drug combinations. CPA allows:

* Out-of-distribution predictions of unseen drug and gene combinations at various doses and among different cell types.
* Learn interpretable drug and cell-type latent spaces.
* Estimate the dose-response curve for each perturbation and their combinations.
* Transfer pertubration effects from on cell-type to an unseen cell-type.
* Enable batch effect removal on a latent space and also gene expression space.


Getting started
***************
To get started with CPA, please follow the installation instructions in the :ref:`installation` section.

Support and contribute
**********************
If you have a question or new architecture or a model that could be integrated into our pipeline, you can
post an `issue <https://github.com/theislab/cpa/issues/new>`_.


.. toctree::
   :maxdepth: 1
   :hidden:

   installation
   api/index
   tutorials/index
   release_notes/index
   references
