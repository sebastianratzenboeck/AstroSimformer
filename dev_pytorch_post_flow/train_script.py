import os
import pandas as pd
import numpy as np
import torch
from domain_alignment import MultiContextFlowModel, DomainAdaptiveTrainer
from dataset import DomainAdaptationDataset, get_train_val_loader
from torch.utils.data import random_split, DataLoader, TensorDataset
import wandb

def main(train_loader, val_loader, config):
    # Initialize Weights & Biases
    wandb.init(
        project = config["project_name"],
        name = config["run_name"],
        config = config,
    )
    # -------------------
    # 3) Instantiate model
    # -------------------
    model = MultiContextFlowModel(
        modalities=config['modalities'],
        num_nodes=config['num_nodes'],
        dim_values=config['dim_values'],
        dim_ids=config['dim_ids'],
        dim_locals=config['dim_locals'],
        dim_globals=config['dim_globals'],
        attn_embed_dims=config['attn_embed_dims'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        widening_factors=config['widening_factors'],
        dropout=config['dropout'],
        dim_theta=config['dim_theta'],
        theta_embed_dim=config['theta_embed_dim'],
        flow_hidden_dim=config['flow_hidden_dim'],
        flow_depth=config['flow_depth'],
        final_embed_dim=config['final_embed_dim'],
        final_num_heads=config['final_num_heads'],
        final_num_layers=config['final_num_layers'],
        final_widening=config['final_widening'],
        mod_value_dim=config['mod_value_dim'],
        mod_id_dim=config['mod_id_dim'],
        mod_local_dim=config['mod_local_dim'],
        mod_global_dim=config['mod_global_dim'],
    ).to(config['device'])
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    # -------------------
    trainer = DomainAdaptiveTrainer(
        model,
        optimizer,
        sigma_min=config['sigma_min'],
        time_prior_exponent=config['time_prior_exponent'],
        device=config['device'],
        # Other kwargs
        ot_w_final=config['ot_w_final'],  # final‐context OT weight
        ot_w_mod=config['ot_w_mod'],  # per-modality OT weight
        pair_weight=config['pair_weight'],
        warmup_steps=config['warmup_steps'],  # how many steps to ramp OT+pair from 0→full
        clip_grad_norm=config['clip_grad_norm'],
        ot_blur_start=config['ot_blur_start'],
        ot_blur_end=config['ot_blur_end'],
        ot_blur_min=config['ot_blur_min'],
        # OT parameters
        scaling=config['scaling'],
        reach=config['reach'],  # “maximum meaningful distance” in your embedding space
        diameter=config['diameter'],  # same as reach here
        truncate=config['truncate']  # optional: cap # of Sinkhorn iterations
    )
    print("Model and trainer initialized.")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")
    print(f"Run id: {wandb.run.id}")
    try:
        # ------------------
        # 4) Training loop
        # ------------------
        best_val_loss = float('inf')
        for epoch in range(1, config['num_epochs'] + 1):
            # Train
            train_stats = trainer.train_epoch(train_loader, epoch, config['num_epochs'])
            # Validate
            val_stats = trainer.eval_epoch(val_loader)

            wandb.log(
                {f"train/{k}": v for k, v in train_stats.items()} |
                {f"val/{k}": v for k, v in val_stats.items()},
                step=epoch
            )
            print(f"Epoch {epoch:3d} | "
                  # {'total': 0., 'flow': 0., 'ot_mod': 0., 'ot_final': 0, 'pair': 0}
                  # f"Train loss: {train_stats['total']:.3f} ▶ flow={train_stats['flow']:.4f} " #, ot_mod={train_stats['ot_mod']:.4f} "
                  f"Train loss: {train_stats['total']:.3f} "  # ▶ flow={train_stats['flow']:.4f} " #, ot_mod={train_stats['ot_mod']:.4f} "
                  # f"pair={train_stats['pair']:.4f}  | "
                  f"Val loss: {val_stats['total']:.3f}")  # ▶ flow={val_stats['flow']:.4f}, ot_mod={val_stats['ot_mod']:.4f} "
            # f"pair={val_stats['pair']:.4f}")

            # Checkpoint on OT metric
            if val_stats['total'] < best_val_loss:
                best_val_loss = val_stats['total']
                ckpt_path = f"checkpoints/{wandb.run.id}_best.pt"
                torch.save(model.state_dict(), ckpt_path)
                # log checkpoint as artifact
                artifact = wandb.Artifact("best-checkpoint", type="model")
                artifact.add_file(ckpt_path)
                wandb.log_artifact(artifact)
                print(f"  → New best validation model saved.")
        # Finish logging
        wandb.finish()
    except:
        wandb.finish()
    print(f'Training complete! Model saved to {ckpt_path}')
    return

if __name__ == "__main__":
    np.random.seed(42)
    torch.set_default_device('cuda')
    # ———————— load data ————————
    fname_basel = '/n/home12/sratzenboeck/data_local/mock/cluster_field_1kpc_fundamentals_Bayestar2019_SED_BaSeL_June2025.parquet'
    fname_btsettl = '/n/home12/sratzenboeck/data_local/mock/cluster_field_1kpc_fundamentals_Edenhofer_SED_BTSettl_June2025_diffClusters.parquet'
    df_load = pd.read_parquet(fname_basel)
    df_load['dist'] = 1000. / df_load['parallax']
    df_real = pd.read_parquet(fname_btsettl)
    df_real['dist'] = 1000. / df_real['parallax']
    N = df_load.shape[0]
    # ———————— define features and modalities ————————
    feats_theta = [
        'mass', 'logAge', 'Z_metal', 'Av', 'parallax'
    ]
    use_log10_preprocessing = {
        'mass': True,
        'logAge': False,
        'Z_metal': False,
        'Av': False,
        'parallax': True,
        'parallax_obs': True
    }

    for col, log_proc in use_log10_preprocessing.items():
        if log_proc:
            df_load[col] = np.log10(
                df_load[col] + np.clip(np.random.normal(1e-3, 1e-3, df_load.shape[0]), a_min=1e-4, a_max=None))
            df_real[col] = np.log10(
                df_real[col] + np.clip(np.random.normal(1e-3, 1e-3, df_real.shape[0]), a_min=1e-4, a_max=None))

    lamost_cols = [col for col in df_load.columns if 'lamost_' in col]
    apogee_cols = [col for col in df_load.columns if 'apogee_' in col]
    boss_cols = [col for col in df_load.columns if 'boss_' in col]
    xp_cols = [col for col in df_load.columns if 'xp_' in col]

    gaia_cols = ['gaia_bp_phot', 'gaia_rp_phot', 'gaia_g_phot', 'parallax_obs']
    tmass_cols = ['2mass_h', '2mass_j', '2mass_k']
    # spitzer_cols = ['irac_1', 'irac_2', 'irac_3', 'irac_4']
    wise_cols = ['wise_1', 'wise_2', 'wise_3', 'wise_4']
    phot_cols = [gaia_cols, tmass_cols, wise_cols]
    # ———————— split data into train and test ————————
    idx_test = np.random.choice(N, 5_000, replace=False)
    idx_all = np.arange(N)
    idx_train = np.asarray(list(set(idx_all) - set(idx_test)))
    # ———————— create dataframes for train and test sets ————————
    df_sim = df_load.loc[idx_train].reset_index(drop=True)
    df_test = df_load.loc[idx_test].reset_index(drop=True)
    # ———————— create idx2match for both dataframes ————————
    same_idx = np.intersect1d(df_real['idx2match'].values, df_sim['idx2match'].values)
    df_sim_sub = df_sim.loc[df_sim.idx2match.isin(same_idx)]
    df_real_sub = df_real.loc[df_real.idx2match.isin(same_idx)]
    df_sim_sub = df_sim_sub.set_index('idx2match').reindex(same_idx).reset_index()
    df_real_sub = df_real_sub.set_index('idx2match').reindex(same_idx).reset_index()
    # ———————— configure input columns and model parameters ————————
    input_cols = {
        'xp': xp_cols,
        'lamost': lamost_cols,
        'boss': boss_cols,
        'apogee': apogee_cols,
        'gaia_phot': gaia_cols,
        '2mass_phot': tmass_cols,
        'wise_phot': wise_cols,
    }
    # 1) Define your two modality groups
    photometric_mods = ['gaia_phot', '2mass_phot', 'wise_phot']  # , 'panstarrs', 'decaps']
    spectral_mods = ['xp']  # , 'lamost'] # 'apogee', 'boss']
    modalities = photometric_mods + spectral_mods
    # 2) Per‐modality feature‐counts
    #    – photometric: only a handful of bands each
    #    – spectral: up to ~7k pixels
    num_nodes = {
        'xp': len(xp_cols),
        'lamost': len(lamost_cols),
        'boss': len(boss_cols),
        'apogee': len(apogee_cols),
        'gaia_phot': len(gaia_cols),
        '2mass_phot': len(tmass_cols),
        'wise_phot': len(wise_cols),
        # 'spitzer_phot': len(spitzer_cols),
    }
    # 3) ContextTransformer hyperparams
    dim_values = {
        **{mod: 20 for mod in photometric_mods},  # small value‐embed
        **{mod: 20 for mod in spectral_mods}      # possibly richer embedding for spectra
    }
    dim_ids = {mod: 20 for mod in modalities}
    dim_locals = {mod: 8 for mod in modalities}
    dim_globals = {mod: 20 for mod in modalities}
    # 4) Attention / Transformer sizes
    attn_embed_dims = {
        **{mod: 32 for mod in photometric_mods},   # lightweight for photometry
        **{mod: 128 for mod in spectral_mods}      # more capacity for spectra
    }
    num_heads = {
        **{mod: 4 for mod in photometric_mods},
        **{mod: 4 for mod in spectral_mods}
    }
    num_layers = {
        **{mod: 3 for mod in photometric_mods},
        **{mod: 3 for mod in spectral_mods}
    }
    widening_factors = {
        **{mod: 4 for mod in photometric_mods},
        **{mod: 4 for mod in spectral_mods}
    }
    # ———————— set hyperparameters ————————
    # Global flow & fusion hyperparameters
    os.makedirs(os.path.dirname('checkpoints/'), exist_ok=True)
    config = dict(
        project_name = "astro-simformer",
        run_name = "multimodal_all_transformers_full_domain_adapt_loss_schedule",
        # model / data
        modalities      = modalities,
        num_nodes       = num_nodes,  # fill in per-mod
        dim_values      = dim_values,
        dim_ids         = dim_ids,
        dim_locals      = dim_locals,
        dim_globals     = dim_globals,
        attn_embed_dims = attn_embed_dims,
        num_heads       = num_heads,
        num_layers      = num_layers,
        widening_factors= widening_factors,
        dropout         = 0.,
        dim_theta       = len(feats_theta),
        theta_embed_dim = 32,
        flow_hidden_dim = 256,
        flow_depth      = 2,
        final_embed_dim = 128,
        final_num_heads = 4,
        final_num_layers= 4,
        final_widening  = 4,
        mod_value_dim   = 64,
        mod_id_dim      = 10,
        mod_local_dim   = 4,
        mod_global_dim  = 12,
        # training
        batch_size      = 1024,
        lr              = 1e-3,
        num_epochs      = 1_000,
        # Domain alignment trainer params
        sigma_min       = 0.,
        time_prior_exponent=1.,
        device          ='cuda' if torch.cuda.is_available() else 'cpu',
        # Other kwargs
        ot_w_final      = 0.1,    # final‐context OT weight
        ot_w_mod        = 0,      # per-modality OT weight
        pair_weight     = 0.1,    # Pair weight
        warmup_steps    = 30_000, # how many steps to ramp OT+pair from 0→full
        clip_grad_norm  = 1.0,
        ot_blur_start   = 1.0,
        ot_blur_end     = 0.1,
        ot_blur_min     = 0.2,
        # OT parameters
        scaling         = 0.5,
        reach           = 2.0,     # “maximum meaningful distance” in your embedding space
        diameter        = 2.0,         # same as reach here
        truncate        = 100          # optional: cap # of Sinkhorn iterations
    )
    gen = torch.Generator(device=config['device']).manual_seed(42)

    train_loader, val_loader, normalization_info = get_train_val_loader(
        df_sim, df_real, df_sim_sub, df_real_sub, input_cols, feats_theta, modalities,
        train_frac=0.9,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=30,
        generator=gen,
        pin_memory=True,
    )
    main(train_loader, val_loader, config)



