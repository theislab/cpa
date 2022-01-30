import anndata
import numpy as np
import pandas as pd
import torch

from scvi.data import setup_anndata
from cpa import CPA
from cpa import register_dataset


def generate_synth():
    n_cells = 2000
    n_genes = 500
    X = np.random.randint(low=0, high=1000, size=(n_cells, n_genes))
    obs = pd.DataFrame(
        dict(
            c0=np.random.randn(n_cells),
            c1=np.random.randn(n_cells),
            drug_name=np.array(["d1", "d2", "d3", "d4", "d5"])[np.random.randint(5, size=n_cells)],
            dose_val=np.array([0.1, 0.05, 0.5, 0.25, 0.75])[np.random.randint(5, size=n_cells)],
            covar_1=np.array(["v1", "v2"])[np.random.randint(2, size=n_cells)],
            covar_2=np.random.randint(10, size=n_cells),
        )
    )
    obs.loc[:, "covar_1"] = obs.loc[:, "covar_1"].astype("category")
    obs.loc[:, "covar_2"] = obs.loc[:, "covar_2"].astype("category")
    dataset = anndata.AnnData(
        X=X,
        obs=obs,
    )

    setup_anndata(dataset)
    drug_encoder, covars_to_ncovars = register_dataset(
        dataset,
        drug_key="drug_name",
        dose_key='dose_val',
        covars_keys=["covar_1", "covar_2"],
    )

    return dict(dataset=dataset, drug_encoder=drug_encoder, covars_to_ncovars=covars_to_ncovars)


def test_cpa():
    data = generate_synth()
    dataset = data["dataset"]
    covars_to_ncovars = data["covars_to_ncovars"]
    model = CPA(
        adata=dataset,
        covars_to_ncovars=covars_to_ncovars,
        loss_ae="gauss",
        variational=False,
    )
    model.train(max_epochs=3, plan_kwargs=dict(lr=1e-4))
    model.predict()

    keys = np.array(["test", "train", "ood"])
    dataset.obs.loc[:, "split"] = keys[
        np.random.randint(0, 3, size=(dataset.shape[0],))
    ]
    model = CPA(
        adata=dataset,
        n_latent=128,
        covars_to_ncovars=covars_to_ncovars,
        loss_ae="gauss",
        doser_type='logsigm',
        split_key="split",
    )
    model.train(max_epochs=3, plan_kwargs=dict(lr=1e-4))

    # model.get_reconstruction_error(adata=dataset, indices=[1, 2, 3, 4])
    # model.get_latent_representation()