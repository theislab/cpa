from cpa import run_autotune
import cpa

import scanpy as sc
from ray import tune
import numpy as np


DATA_PATH = '/PATH/TO/DATA.h5ad' # Change this to your desired path
adata = sc.read_h5ad(DATA_PATH)
adata.X = adata.layers['counts'].copy()  # Counts should be available in the 'counts' layer
sc.pp.subsample(adata, fraction=0.1)

model_args = {
    'n_latent': tune.choice([32, 64, 128, 256]),
    'recon_loss': tune.choice(['nb']),
    'doser_type': tune.choice(['logsigm']),

    'n_hidden_encoder': tune.choice([128, 256, 512, 1024]),
    'n_layers_encoder': tune.choice([1, 2, 3, 4, 5]),

    'n_hidden_decoder': tune.choice([128, 256, 512, 1024]),
    'n_layers_decoder': tune.choice([1, 2, 3, 4, 5]),

    'use_batch_norm_encoder': tune.choice([True, False]),
    'use_layer_norm_encoder': tune.sample_from(
        lambda spec: False if spec.config.model_args.use_batch_norm_encoder else np.random.choice([True, False])),

    'use_batch_norm_decoder': tune.choice([True, False]),
    'use_layer_norm_decoder': tune.sample_from(
        lambda spec: False if spec.config.model_args.use_batch_norm_decoder else np.random.choice([True, False])),

    'dropout_rate_encoder': tune.choice([0.0, 0.1, 0.2, 0.25]),
    'dropout_rate_decoder': tune.choice([0.0, 0.1, 0.2, 0.25]),

    'variational': tune.choice([False]),

    'seed': tune.randint(0, 10000),
    
    'split_key': tune.choice(['split_1ct_MEC']),
    'train_split': tune.choice(['train']),
    'valid_split': tune.choice(['valid']),
    'test_split': tune.choice(['ood']),
}

train_args = {
    'n_epochs_adv_warmup': tune.choice([0, 1, 3, 5, 10, 50, 70]),
    'n_epochs_kl_warmup': tune.choice([None]),
        # lambda spec: None if not spec.config.model_args.variational else np.random.choice([0, 1, 3, 5, 10])),

    'n_epochs_pretrain_ae': tune.choice([0, 1, 3, 5, 10, 30, 50]),

    'adv_steps': tune.choice([2, 3, 5, 10, 15, 20, 25, 30]),

    'mixup_alpha': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
    'n_epochs_mixup_warmup': tune.sample_from(
        lambda spec: 0 if spec.config.train_args.mixup_alpha == 0.0 else np.random.choice([
            0, 1, 3, 5, 10]),
    ),

    'n_layers_adv': tune.choice([1, 2, 3, 4, 5]),
    'n_hidden_adv': tune.choice([32, 64, 128, 256]),

    'use_batch_norm_adv': tune.choice([True, False]),
    'use_layer_norm_adv': tune.sample_from(
        lambda spec: False if spec.config.train_args.use_batch_norm_adv else np.random.choice([True, False])),

    'dropout_rate_adv': tune.choice([0.0, 0.1, 0.2, 0.25, 0.3]),

    'pen_adv': tune.loguniform(1e-2, 1e2),
    'reg_adv': tune.loguniform(1e-2, 1e2),

    'lr': tune.loguniform(1e-5, 1e-2),
    'wd': tune.loguniform(1e-8, 1e-5),

    'doser_lr': tune.loguniform(1e-5, 1e-2),
    'doser_wd': tune.loguniform(1e-8, 1e-5),

    'adv_lr': tune.loguniform(1e-5, 1e-2),
    'adv_wd': tune.loguniform(1e-8, 1e-5),

    'adv_loss': tune.choice(['cce']),

    'do_clip_grad': tune.choice([True, False]),
    # 'gradient_clip_value': tune.loguniform(1e-2, 1e2),
    'gradient_clip_value': tune.choice([1.0]),

    'step_size_lr': tune.choice([10, 25, 45]),
}

search_space = {
    'model_args': model_args,
    'train_args': train_args,
}

setup_anndata_kwargs = {
            'perturbation_key': 'condition_ID',
            'dosage_key': 'log_dose',
            'control_group': 'CHEMBL504',
            'batch_key': None,
            'is_count_data': True,
            'categorical_covariate_keys': ['cell_type'],
            'deg_uns_key': 'rank_genes_groups_cov',
            'deg_uns_cat_key': 'cov_drug_dose',
            'max_comb_len': 2,
}
model = cpa.CPA
model.setup_anndata(adata, **setup_anndata_kwargs)

run_autotune(
    model_cls=model,
    data=adata,
    metrics=["cpa_metric",  # The first one (cpa_metric) is the one that will be used for optimization "MAIN ONE"
             "disnt_basal",
             "disnt_after",
             "r2_mean",
             "r2_var",
             "r2_mean_lfc",
             "r2_var_lfc",
             "recon_loss",
             "KL"],
    mode="max",
    search_space=search_space,
    num_samples=5000,  # Change this to your desired number of samples (Number of runs)
    scheduler="asha",
    searcher="hyperopt",
    seed=1,
    resources={"cpu": 40, "gpu":0.2, "memory": 16000},
    experiment_name="cpa_autotune",  # Change this to your desired experiment name
    logging_dir='/PATH/TO/LOGS/',  # Change this to your desired path
    adata_path=DATA_PATH,
    sub_sample=0.1,
    setup_anndata_kwargs=setup_anndata_kwargs,
    use_wandb=True,  # If you want to use wandb, set this to True
    wandb_name="cpa_tune",  # Change this to your desired wandb project name
    # scheduler_kwargs: dict | None = None,
    # searcher_kwargs: dict | None = None,
)