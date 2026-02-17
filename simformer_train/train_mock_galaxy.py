#!/usr/bin/env python
"""
Train SimFormer on galaxy photometric data.

Reads a Parquet/CSV file that already has _obs, _err columns and NaN
patterns (produced by prepare_data.py for mock data, or directly from
real survey data). Handles normalization, model creation, curriculum
scheduling, and training.

Features:
  - Curriculum data scheduling: temperature τ ramps from 0 (uniform age
    sampling) to τ_max (closer to natural age distribution) over training.
  - Cosine annealing LR schedule.
  - Mixed-precision training (AMP) and torch.compile support.
  - Observed/unobserved mask embeddings.

Usage:
  python train_mock_galaxy.py --data-path prepared_data.parquet --epochs 50
  python train_mock_galaxy.py --data-path prepared_data.parquet --epochs 1000 \\
      --tau-warmup 100 --tau-max 0.7 --amp --compile --wandb --device cuda
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from columns import (
    INTRINSIC_COLS, TRUE_MAG_COLS, OBS_COLS, OBS_ERR_COLS,
    ALL_VALUE_COLS, NUM_NODES, N_INTRINSIC, N_TRUE_MAG,
)
from transformer import Simformer
from simflower import FlowMatchingTrainer
from utils import make_condition_mask_generator


# ---------------------------------------------------------------------------
# Data loading & array building
# ---------------------------------------------------------------------------
def load_data(data_path):
    """Load a Parquet or CSV file with expected column structure."""
    print(f'Loading data from {data_path} ...')
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)
    print(f'  Loaded {len(df):,} stars, {len(df.columns)} columns')
    return df


def build_arrays(df):
    """Build value, error, and observed mask arrays from DataFrame.

    Returns:
        values_norm: (N, NUM_NODES) normalized values, NaN filled with 0
        errors_norm: (N, NUM_NODES) standardized log-errors with sentinels:
            -5 for perfectly known (error=0), +5 for unobserved (NaN),
            ~N(0,1) for real measurements
        observed_mask: (N, NUM_NODES) binary mask (1=observed, 0=unobserved)
        means: (NUM_NODES,) per-column means (ignoring NaN)
        stds: (NUM_NODES,) per-column stds (ignoring NaN)
        log_err_mean: scalar, mean of log(real errors) for denormalization
        log_err_std: scalar, std of log(real errors) for denormalization
    """
    values_raw = df[ALL_VALUE_COLS].values.astype(np.float32)

    # Error array: 0 for intrinsic/true mag cols, actual errors for obs cols
    errors_raw = np.zeros_like(values_raw)
    for i, err_col in enumerate(OBS_ERR_COLS):
        col_idx = N_INTRINSIC + N_TRUE_MAG + i
        errors_raw[:, col_idx] = df[err_col].values.astype(np.float32)

    # Observed mask: 1 for intrinsic/true mag (always observed), 0/1 for obs cols
    observed_mask = np.ones_like(values_raw)
    for i, obs_col in enumerate(OBS_COLS):
        col_idx = N_INTRINSIC + N_TRUE_MAG + i
        observed_mask[:, col_idx] = (~df[obs_col].isna()).values.astype(np.float32)

    # Normalize values
    means = np.nanmean(values_raw, axis=0)
    stds = np.nanstd(values_raw, axis=0)
    stds[stds < 1e-10] = 1.0

    values_norm = (values_raw - means) / stds
    values_norm[np.isnan(values_norm)] = 0.0

    # Log-transform + standardize errors with sentinels
    # Three regimes in normalized log-error space:
    #   perfectly known (error=0, e.g. intrinsic cols) → -5
    #   real measurements (error>0, not NaN)           → ~N(0,1)
    #   unobserved (NaN error)                         → +5
    LOG_ERR_PERFECT = -5.0
    LOG_ERR_UNOBS = 5.0

    has_real_error = (errors_raw > 0) & ~np.isnan(errors_raw)
    is_zero_error = (errors_raw == 0)

    log_errors_real = np.log(errors_raw[has_real_error])
    log_err_mean = float(log_errors_real.mean())
    log_err_std = float(log_errors_real.std())
    if log_err_std < 1e-10:
        log_err_std = 1.0

    errors_norm = np.full_like(errors_raw, LOG_ERR_UNOBS)
    errors_norm[has_real_error] = (np.log(errors_raw[has_real_error]) - log_err_mean) / log_err_std
    errors_norm[is_zero_error] = LOG_ERR_PERFECT

    print(f'  Arrays: {values_norm.shape[0]:,} stars x {values_norm.shape[1]} nodes')
    print(f'  Unobserved entries: {(observed_mask == 0).sum():,}')
    print(f'  Log-error stats: mean={log_err_mean:.3f}, std={log_err_std:.3f}')
    print(f'  Error regimes: perfect={LOG_ERR_PERFECT}, real~N(0,1), unobs={LOG_ERR_UNOBS}')
    return values_norm, errors_norm, observed_mask, means, stds, log_err_mean, log_err_std


# ---------------------------------------------------------------------------
# Curriculum scheduling
# ---------------------------------------------------------------------------
def compute_age_bin_indices(log_age, n_bins):
    """Assign each star to an age bin. Returns bin indices and bin counts."""
    bin_edges = np.linspace(log_age.min(), log_age.max() + 1e-6, n_bins + 1)
    bin_idx = np.digitize(log_age, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    bin_counts = np.bincount(bin_idx, minlength=n_bins).astype(np.float64)
    return bin_idx, bin_counts


def compute_tau(epoch, total_epochs, tau_max, tau_warmup):
    """Compute temperature τ for the current epoch.

    τ=0 during warmup (uniform sampling), then linearly ramps to τ_max.
    """
    if epoch < tau_warmup:
        return 0.0
    ramp_epochs = total_epochs - tau_warmup
    if ramp_epochs <= 0:
        return tau_max
    progress = (epoch - tau_warmup) / ramp_epochs
    return tau_max * min(progress, 1.0)


def build_epoch_indices(bin_idx, bin_counts, tau, cap_per_bin, rng=None):
    """Select unique star indices for one epoch via capped per-bin sampling.

    At τ=0 every bin contributes min(count, cap_per_bin) stars (uniform across
    bins).  As τ→1 larger bins contribute proportionally more, approaching
    the natural distribution.  Each star appears at most once per epoch.

    Args:
        bin_idx:      1-D array, bin assignment for every star in this split.
        bin_counts:   1-D array, number of stars per bin in this split.
        tau:          Temperature (0 = uniform across bins, 1 = natural).
        cap_per_bin:  Base cap at τ=0.  Larger bins may exceed this when τ>0.
        rng:          numpy random Generator (for reproducibility).

    Returns:
        1-D numpy array of shuffled global indices (into train or val data).
    """
    if rng is None:
        rng = np.random.default_rng()

    nonzero = bin_counts[bin_counts > 0]
    min_count = nonzero.min()

    selected = []
    for b, count in enumerate(bin_counts):
        count = int(count)
        if count == 0:
            continue
        ratio = (count / min_count) ** tau          # 1.0 at τ=0, grows with τ
        n_select = min(count, max(1, round(cap_per_bin * ratio)))
        members = np.where(bin_idx == b)[0]
        chosen = rng.choice(members, size=n_select, replace=False)
        selected.append(chosen)

    indices = np.concatenate(selected)
    rng.shuffle(indices)
    return indices


def make_epoch_callback(bin_idx_train, bin_counts_train,
                        bin_idx_val, bin_counts_val,
                        tau_max, tau_warmup,
                        cap_per_bin=1000,
                        use_wandb=False):
    """Create the epoch callback that rebuilds train/val index arrays each epoch.

    Uses cap-based per-bin sampling (without replacement) instead of weighted
    multinomial sampling.  At τ=0 every bin contributes up to cap_per_bin
    unique stars; as τ grows, larger bins contribute proportionally more.
    """
    rng = np.random.default_rng(42)

    def epoch_callback(trainer, epoch, total_epochs):
        tau = compute_tau(epoch, total_epochs, tau_max, tau_warmup)

        train_indices = build_epoch_indices(
            bin_idx_train, bin_counts_train, tau, cap_per_bin, rng)
        val_indices = build_epoch_indices(
            bin_idx_val, bin_counts_val, tau, cap_per_bin, rng)

        trainer.set_epoch_indices(train_indices)
        trainer.set_val_epoch_indices(val_indices)

        n_train_steps = len(train_indices) // trainer.batch_size
        n_val_steps = len(val_indices) // trainer.batch_size
        current_lr = trainer.optimizer.param_groups[0]['lr']
        print(f'  [Curriculum] epoch={epoch+1}, τ={tau:.3f}, '
              f'train_stars={len(train_indices):,} ({n_train_steps} steps), '
              f'val_stars={len(val_indices):,} ({n_val_steps} steps), '
              f'lr={current_lr:.2e}')

        if use_wandb:
            import wandb
            wandb.log({'tau': tau, 'learning_rate': current_lr,
                       'epoch': epoch + 1,
                       'train_stars': len(train_indices),
                       'val_stars': len(val_indices)})

    return epoch_callback


# ---------------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------------
def create_model():
    """Create SimFormer model with default architecture."""
    model = Simformer(
        num_nodes=NUM_NODES,
        dim_value=24,
        dim_id=24,
        dim_condition=16,
        dim_error=16,
        dim_observed=8,
        attn_embed_dim=128,
        num_heads=8,
        num_layers=4,
        widening_factor=4,
        time_embed_dim=64,
        dropout=0.05,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Model created: {n_params:,} parameters')
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description='Train SimFormer on galaxy data with curriculum scheduling.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    p.add_argument('--data-path', type=str, required=True,
                   help='Path to prepared Parquet/CSV with _obs/_err columns')
    p.add_argument('--output-dir', type=str, default='./output',
                   help='Directory for checkpoints and normalization stats')

    # Training
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=1e-3,
                   help='Initial learning rate')
    p.add_argument('--lr-min', type=float, default=1e-5,
                   help='Minimum LR for cosine annealing')
    p.add_argument('--inner-loop-size', type=int, default=500,
                   help='Training steps per epoch')
    p.add_argument('--patience', type=int, default=20,
                   help='Early stopping patience (epochs)')
    p.add_argument('--val-split', type=float, default=0.15)
    p.add_argument('--dense-ratio', type=float, default=0.8,
                   help='Fraction of batch with fully connected edge masks (rest get random sparsity)')

    # Curriculum scheduling
    p.add_argument('--n-bins', type=int, default=30,
                   help='Number of age bins for curriculum weighting')
    p.add_argument('--tau-max', type=float, default=0.8,
                   help='Max temperature (0=uniform, 1=natural distribution)')
    p.add_argument('--tau-warmup', type=int, default=10,
                   help='Epochs to stay at τ=0 before ramping')
    p.add_argument('--cap-per-bin', type=int, default=1000,
                   help='Max unique stars per age bin at τ=0 (bins with fewer stars contribute all)')

    # Performance
    p.add_argument('--amp', action='store_true', default=False,
                   help='Enable mixed-precision training (FP16). Recommended for GPU.')
    p.add_argument('--grad-clip-norm', type=float, default=1.0,
                   help='Max gradient norm for clipping (stabilizes training)')
    p.add_argument('--weight-decay', type=float, default=1e-4,
                   help='AdamW weight decay for regularization')
    p.add_argument('--compile', action='store_true', default=False,
                   help='Use torch.compile for kernel fusion (PyTorch 2.x)')

    # WandB
    p.add_argument('--wandb', action='store_true', default=False,
                   help='Enable WandB logging')
    p.add_argument('--wandb-project', type=str, default='mock-galaxy-simformer')

    # Device / seed
    p.add_argument('--device', type=str, default=None,
                   help='Device (auto-detect if not set)')
    p.add_argument('--seed', type=int, default=42)

    return p.parse_args()


def main():
    args = parse_args()

    # Seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f'Using device: {device}')

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load data ----
    print('\n--- Data ---')
    df = load_data(args.data_path)
    data, data_errors, data_observed_mask, means, stds, log_err_mean, log_err_std = build_arrays(df)

    # Save normalization stats
    norm_path = os.path.join(args.output_dir, 'norm_stats.npz')
    np.savez(norm_path, means=means, stds=stds, columns=ALL_VALUE_COLS,
             log_err_mean=log_err_mean, log_err_std=log_err_std)
    print(f'  Normalization stats saved to {norm_path}')

    # ---- Age bin indices for curriculum weighting ----
    # We need bin indices that correspond to the training split.
    # The trainer does train_test_split internally, so we replicate the split
    # to get the correct bin indices for the training subset.
    log_age = df['logAge'].values
    bin_idx_all, _ = compute_age_bin_indices(log_age, args.n_bins)

    from sklearn.model_selection import train_test_split
    n_total = len(data)
    all_indices = np.arange(n_total)
    train_indices, val_indices = train_test_split(
        all_indices, test_size=args.val_split, random_state=42
    )
    bin_idx_train = bin_idx_all[train_indices]
    bin_idx_val = bin_idx_all[val_indices]
    bin_counts_train = np.bincount(bin_idx_train, minlength=args.n_bins).astype(np.float64)
    bin_counts_val = np.bincount(bin_idx_val, minlength=args.n_bins).astype(np.float64)
    print(f'  Curriculum: {args.n_bins} bins, train={len(train_indices):,}, val={len(val_indices):,}')

    # ---- Model ----
    print('\n--- Model ---')
    model = create_model()
    if args.compile:
        print('  Compiling model with torch.compile ...')
        model = torch.compile(model)

    # ---- Condition mask generator ----
    obs_indices = list(range(N_INTRINSIC + N_TRUE_MAG, NUM_NODES))
    cond_gen = make_condition_mask_generator(
        batch_size=args.batch_size,
        num_features=NUM_NODES,
        percent=(0.1, 0.5),
        allowed_idx=obs_indices,
        device=device,
    )

    # ---- Trainer ----
    print('\n--- Trainer ---')
    trainer = FlowMatchingTrainer(
        model=model,
        data=data,
        data_errors=data_errors,
        data_observed_mask=data_observed_mask,
        condition_mask_generator=cond_gen,
        batch_size=args.batch_size,
        lr=args.lr,
        inner_train_loop_size=args.inner_loop_size,
        early_stopping_patience=args.patience,
        val_split=args.val_split,
        dense_ratio=args.dense_ratio,
        use_amp=args.amp,
        grad_clip_norm=args.grad_clip_norm,
        weight_decay=args.weight_decay,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_config=vars(args),
        device=device,
    )
    print('  Trainer initialized.')

    # ---- LR scheduler ----
    lr_scheduler = CosineAnnealingLR(
        trainer.optimizer, T_max=args.epochs, eta_min=args.lr_min
    )

    # ---- Epoch callback (curriculum) ----
    epoch_cb = make_epoch_callback(
        bin_idx_train=bin_idx_train,
        bin_counts_train=bin_counts_train,
        bin_idx_val=bin_idx_val,
        bin_counts_val=bin_counts_val,
        tau_max=args.tau_max,
        tau_warmup=args.tau_warmup,
        cap_per_bin=args.cap_per_bin,
        use_wandb=args.wandb,
    )

    # ---- Train ----
    print(f'\n--- Training ({args.epochs} epochs) ---')
    t0 = time.time()
    best_model = trainer.fit(
        epochs=args.epochs,
        verbose=True,
        epoch_callback=epoch_cb,
        lr_scheduler=lr_scheduler,
    )
    elapsed = time.time() - t0
    print(f'\nTraining completed in {elapsed / 60:.1f} minutes.')

    # ---- Save model ----
    ckpt_path = os.path.join(args.output_dir, 'best_model.pt')
    torch.save(best_model.state_dict(), ckpt_path)
    print(f'Best model saved to {ckpt_path}')

    # ---- Summary ----
    print('\n--- Summary ---')
    print(f'  Data:       {data.shape[0]:,} stars, {data.shape[1]} nodes')
    print(f'  Model:      {sum(p.numel() for p in best_model.parameters()):,} params')
    print(f'  Epochs:     {args.epochs}')
    print(f'  Curriculum: τ_warmup={args.tau_warmup}, τ_max={args.tau_max}')
    print(f'  LR:         {args.lr} → {args.lr_min} (cosine)')
    print(f'  AMP:        {args.amp}')
    print(f'  Compiled:   {args.compile}')
    print(f'  Output:     {args.output_dir}/')


if __name__ == '__main__':
    main()
