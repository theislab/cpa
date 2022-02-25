import numpy as np
import scanpy as sc
from anndata import AnnData
from scvi.data import register_tensor_from_anndata
from sklearn.preprocessing import OneHotEncoder

from ._model import CPA
from ._utils import _CE_CONSTANTS


def prepare_dataset(
        data_path,
        drug_key,
        dose_key,
        covars_keys,
        subsample=False,
        combinatorial=False,
):
    """TEMPORARY.

    Quick and dirty way to construct the dataloader for the CPA model.
    This function will be replaced once the AnnData refactor is completed within
    scvi-tools.

    Parameters
    ----------
    adata : AnnData
    drug_key : str
        Obs key for the drug names
    dose_key : str
        Obs key for drug doses
    covars_keys : list
        List of categorical covariates
    """
    _CE_CONSTANTS.DRUG_KEY = drug_key
    _CE_CONSTANTS.COVARS_KEYS = covars_keys
    _CE_CONSTANTS.DOSE_KEY = dose_key

    adata = sc.read(data_path)

    if subsample:
        sc.pp.subsample(adata, fraction=0.5)

    CPA.setup_anndata(adata)

    register_tensor_from_anndata(adata, "drug_name", "obs", drug_key, is_categorical=True)
    register_tensor_from_anndata(adata, "dose_value", "obs", dose_key, is_categorical=False)

    if combinatorial:
        drugs = adata.obs[drug_key]

        # get unique drugs
        drugs_names_unique = set()
        for d in drugs:
            [drugs_names_unique.add(i) for i in d.split("+")]
        drugs_names_unique = np.array(list(drugs_names_unique))

        drug_encoder = OneHotEncoder(sparse=False)
        drug_encoder.fit(drugs_names_unique.reshape(-1, 1))

        drugs_doses = []
        for i, comb in enumerate(drugs):
            drugs_combos = drug_encoder.transform(
                np.array(comb.split("+")).reshape(-1, 1))
            dose_combos = str(adata.obs[dose_key].values[i]).split("+")
            for j, d in enumerate(dose_combos):
                if j == 0:
                    drug_ohe = float(d) * drugs_combos[j]
                else:
                    drug_ohe += float(d) * drugs_combos[j]
            drugs_doses.append(drug_ohe)

        adata.obsm['drugs_doses'] = np.array(drugs_doses)
        register_tensor_from_anndata(adata, "drugs_doses", "obsm", "drugs_doses")

    else:
        drug_encoder = {drug: i for i, drug in
                        enumerate(adata.uns['_scvi']['categorical_mappings'][f'{drug_key}_scvi']['mapping'])}

    covars_encoder = {}
    for covar in covars_keys:
        register_tensor_from_anndata(adata, covar, "obs", covar, is_categorical=True)
        covars_encoder[covar] = {covar_value: index for index, covar_value in enumerate(
            list(adata.uns['_scvi']['categorical_mappings'][f'{covar}_scvi']['mapping']))}

    return adata, drug_encoder, covars_encoder
