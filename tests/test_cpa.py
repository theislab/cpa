import anndata
import numpy as np
import pandas as pd

from scvi.data import setup_anndata
import cpa


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
            control=np.random.randint(10, size=n_cells),
            split=np.array(["train", "test", "ood"])[np.random.randint(3, size=n_cells)],
        )
    )
    obs.loc[:, "covar_1"] = obs.loc[:, "covar_1"].astype("category")
    obs.loc[:, "covar_2"] = obs.loc[:, "covar_2"].astype("category")
    obs.loc[:, "control"] = obs.loc[:, "control"].astype("category")

    dataset = anndata.AnnData(
        X=X,
        obs=obs,
    )

    cpa.CPA.setup_anndata(
        dataset,
        drug_key="drug_name",
        dose_key='dose_val',
        categorical_covariate_keys=["covar_1", "covar_2"],
        control_key='control'
    )

    return dict(dataset=dataset)


def test_cpa():
    data = generate_synth()
    dataset = data["dataset"]
    model = cpa.CPA(adata=dataset,
                    n_latent=128,
                    recon_loss='gauss',
                    doser_type='logsigm',
                    split_key='split',
                    )
    model.train(max_epochs=3, plan_kwargs=dict(autoencoder_lr=1e-4))
    model.predict(batch_size=1024)
