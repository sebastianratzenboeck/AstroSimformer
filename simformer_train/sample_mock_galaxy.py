#!/usr/bin/env python
"""
Sample from a trained SimFormer model for galaxy posterior inference.

Given observed photometric data (magnitudes, parallax, etc.), generate
posterior samples for intrinsic stellar parameters (age, metallicity,
mass, luminosity, ...) and true (noise-free) magnitudes.

Usage:
  # Sample posteriors for stars in a CSV/Parquet file
  python sample_mock_galaxy.py --model-dir ./output --obs-file stars.parquet

  # More posterior draws, finer ODE integration, on GPU
  python sample_mock_galaxy.py --model-dir ./output --obs-file stars.parquet \\
      --num-samples 2000 --steps 128 --batch-size 1024 --device cuda

  # Output to a specific file
  python sample_mock_galaxy.py --model-dir ./output --obs-file stars.parquet \\
      --output posteriors.parquet
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

from columns import (
    INTRINSIC_COLS, TRUE_MAG_COLS, OBS_COLS, OBS_ERR_COLS,
    ALL_VALUE_COLS, NUM_NODES, N_INTRINSIC, N_TRUE_MAG,
)
from transformer import Simformer
from inference_utils import NormStats
from sampling import (
    build_inference_edge_mask,
    build_inference_condition_mask,
    build_inference_node_ids,
    sample_batched_flow,
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_dir, device="cpu"):
    """Load a trained SimFormer model and normalization statistics.

    Args:
        model_dir: Directory containing ``best_model.pt`` and ``norm_stats.npz``.
        device: Target device.

    Returns:
        model: Simformer model with loaded weights.
        norm_stats: NormStats instance.
    """
    # Architecture must match create_model() in train_mock_galaxy.py
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

    ckpt_path = os.path.join(model_dir, "best_model.pt")
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    norm_stats = NormStats(os.path.join(model_dir, "norm_stats.npz"))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded: {n_params:,} parameters from {ckpt_path}")
    return model, norm_stats


# ---------------------------------------------------------------------------
# Observation preparation
# ---------------------------------------------------------------------------

def prepare_observations(obs_df, norm_stats, device="cpu"):
    """Convert raw observations into model-ready tensors.

    Each row of *obs_df* represents one star. Columns should match OBS_COLS
    (with NaN for unobserved bands) and optionally OBS_ERR_COLS.

    Intrinsic and true-magnitude columns are left as unobserved (NaN) since
    those are what we want to infer.

    Args:
        obs_df: DataFrame with observed columns (subset of OBS_COLS / OBS_ERR_COLS).
        norm_stats: NormStats instance from training.
        device: Target device.

    Returns:
        condition_values: (N_stars, NUM_NODES) float tensor — normalized,
            NaN filled with 0.
        condition_mask: (N_stars, NUM_NODES, 1) float tensor — 1 for
            conditioned (observed) dims.
        observed_mask: (N_stars, NUM_NODES) float tensor — 1 for observed dims.
        errors: (N_stars, NUM_NODES) float tensor — measurement errors
            (0 where unavailable).
    """
    N_stars = len(obs_df)

    # --- Value array: all NUM_NODES columns ---
    # Intrinsic + true mag = NaN (to be inferred); obs = from DataFrame
    values_raw = np.full((N_stars, NUM_NODES), np.nan, dtype=np.float32)

    for i, col in enumerate(OBS_COLS):
        col_idx = N_INTRINSIC + N_TRUE_MAG + i
        if col in obs_df.columns:
            values_raw[:, col_idx] = obs_df[col].values.astype(np.float32)

    # --- Observed mask ---
    observed_mask = np.zeros((N_stars, NUM_NODES), dtype=np.float32)
    for i, col in enumerate(OBS_COLS):
        col_idx = N_INTRINSIC + N_TRUE_MAG + i
        if col in obs_df.columns:
            observed_mask[:, col_idx] = (~np.isnan(values_raw[:, col_idx])).astype(np.float32)

    # --- Condition mask ---
    # Condition on all observed OBS_COLS (where we have actual measurements)
    condition_mask = observed_mask.copy()  # (N_stars, NUM_NODES)

    # --- Errors array ---
    errors_raw = np.zeros((N_stars, NUM_NODES), dtype=np.float32)
    for i, err_col in enumerate(OBS_ERR_COLS):
        col_idx = N_INTRINSIC + N_TRUE_MAG + i
        if err_col in obs_df.columns:
            err_vals = obs_df[err_col].values.astype(np.float32)
            err_vals = np.nan_to_num(err_vals, nan=0.0)
            errors_raw[:, col_idx] = err_vals

    # --- Normalize values ---
    values_norm = norm_stats.normalize_numpy(values_raw)
    values_norm = np.nan_to_num(values_norm, nan=0.0)

    # --- Convert to tensors ---
    condition_values = torch.tensor(values_norm, dtype=torch.float32, device=device)
    condition_mask = torch.tensor(condition_mask, dtype=torch.float32, device=device).unsqueeze(-1)
    observed_mask = torch.tensor(observed_mask, dtype=torch.float32, device=device)
    errors = torch.tensor(errors_raw, dtype=torch.float32, device=device)

    return condition_values, condition_mask, observed_mask, errors


# ---------------------------------------------------------------------------
# Posterior sampling
# ---------------------------------------------------------------------------

def sample_posterior(
    model,
    condition_values,
    condition_mask,
    observed_mask,
    errors,
    num_samples=512,
    batch_size=512,
    steps=128,
    device="cpu",
):
    """Generate posterior samples for each star.

    For each of the N_stars input observations, draws *num_samples*
    independent posterior samples by running the flow from t=0 to t=1.

    Args:
        model: Trained Simformer model.
        condition_values: (N_stars, NUM_NODES) normalized values.
        condition_mask: (N_stars, NUM_NODES, 1) condition mask.
        observed_mask: (N_stars, NUM_NODES) observed mask.
        errors: (N_stars, NUM_NODES) measurement errors.
        num_samples: Number of posterior draws per star.
        batch_size: Maximum batch size for parallel sampling.
        steps: Number of Euler integration steps.
        device: Target device.

    Returns:
        all_samples: (N_stars, num_samples, NUM_NODES) tensor in normalized space.
    """
    N_stars = condition_values.shape[0]
    M = condition_values.shape[1]
    all_samples = torch.zeros(N_stars, num_samples, M, device="cpu")

    for star_idx in range(N_stars):
        print(f"  Sampling star {star_idx + 1}/{N_stars} "
              f"({num_samples} draws, {steps} steps) ...")

        # Replicate this star's data num_samples times
        cv = condition_values[star_idx].unsqueeze(0).expand(num_samples, -1)  # (S, M)
        cm = condition_mask[star_idx].unsqueeze(0).expand(num_samples, -1, -1)  # (S, M, 1)
        om = observed_mask[star_idx].unsqueeze(0).expand(num_samples, -1)   # (S, M)
        er = errors[star_idx].unsqueeze(0).expand(num_samples, -1)          # (S, M)

        # Process in chunks of batch_size
        star_samples = []
        for chunk_start in range(0, num_samples, batch_size):
            chunk_end = min(chunk_start + batch_size, num_samples)
            B = chunk_end - chunk_start

            cv_chunk = cv[chunk_start:chunk_end].to(device)
            cm_chunk = cm[chunk_start:chunk_end].to(device)
            om_chunk = om[chunk_start:chunk_end].to(device)
            er_chunk = er[chunk_start:chunk_end].to(device)

            # Build inference masks
            node_ids = build_inference_node_ids(B, M, device=device)
            edge_mask = build_inference_edge_mask(B, M, observed_mask=om_chunk, device=device)

            # Run flow
            x = sample_batched_flow(
                model_fn=model,
                shape=(B,),
                condition_mask=cm_chunk,
                condition_values=cv_chunk,
                node_ids=node_ids,
                edge_masks=edge_mask,
                errors=er_chunk,
                observed_mask=om_chunk,
                steps=steps,
                device=device,
            )  # (B, M, 1)

            star_samples.append(x.squeeze(-1).cpu())  # (B, M)

        all_samples[star_idx] = torch.cat(star_samples, dim=0)  # (num_samples, M)

    return all_samples


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def samples_to_dataframe(all_samples, norm_stats, star_ids=None):
    """Convert posterior samples to a long-format DataFrame.

    Args:
        all_samples: (N_stars, num_samples, NUM_NODES) tensor in normalized space.
        norm_stats: NormStats instance for denormalization.
        star_ids: Optional list/array of star identifiers.

    Returns:
        DataFrame with columns: [star_id, sample_idx, <column_names>...]
    """
    N_stars, num_samples, M = all_samples.shape

    # Denormalize to physical units
    samples_phys = norm_stats.denormalize(all_samples.view(-1, M)).view(N_stars, num_samples, M)
    samples_np = samples_phys.numpy()

    rows = []
    for star_idx in range(N_stars):
        sid = star_ids[star_idx] if star_ids is not None else star_idx
        for sample_idx in range(num_samples):
            row = {"star_id": sid, "sample_idx": sample_idx}
            for col_idx, col_name in enumerate(ALL_VALUE_COLS):
                row[col_name] = samples_np[star_idx, sample_idx, col_idx]
            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Sample posteriors from trained SimFormer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--model-dir", type=str, required=True,
                   help="Directory with best_model.pt and norm_stats.npz")
    p.add_argument("--obs-file", type=str, required=True,
                   help="CSV/Parquet with observed columns (_obs, _err)")
    p.add_argument("--num-samples", type=int, default=512,
                   help="Number of posterior draws per star")
    p.add_argument("--steps", type=int, default=128,
                   help="Number of Euler ODE integration steps")
    p.add_argument("--batch-size", type=int, default=512,
                   help="Batch size for parallel sampling")
    p.add_argument("--device", type=str, default=None,
                   help="Device (auto-detect if not set)")
    p.add_argument("--output", type=str, default=None,
                   help="Output file path (default: <model-dir>/posteriors.parquet)")
    p.add_argument("--max-stars", type=int, default=None,
                   help="Limit number of stars to sample (for testing)")

    return p.parse_args()


def main():
    args = parse_args()

    # Device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # ---- Load model ----
    print("\n--- Model ---")
    model, norm_stats = load_model(args.model_dir, device=device)

    # ---- Load observations ----
    print("\n--- Observations ---")
    if args.obs_file.endswith(".parquet"):
        obs_df = pd.read_parquet(args.obs_file)
    else:
        obs_df = pd.read_csv(args.obs_file)

    if args.max_stars is not None:
        obs_df = obs_df.head(args.max_stars)

    print(f"  Loaded {len(obs_df)} stars from {args.obs_file}")

    # Check which observed columns are present
    present_obs = [c for c in OBS_COLS if c in obs_df.columns]
    present_err = [c for c in OBS_ERR_COLS if c in obs_df.columns]
    print(f"  Observed columns found: {len(present_obs)}/{len(OBS_COLS)}")
    print(f"  Error columns found:    {len(present_err)}/{len(OBS_ERR_COLS)}")

    if not present_obs:
        raise ValueError(
            f"No observed columns found in {args.obs_file}. "
            f"Expected columns like: {OBS_COLS[:3]}"
        )

    # ---- Prepare inputs ----
    print("\n--- Preparing inputs ---")
    condition_values, condition_mask, observed_mask, errors = \
        prepare_observations(obs_df, norm_stats, device="cpu")

    # Summary
    n_cond_per_star = condition_mask.squeeze(-1).sum(dim=1)
    print(f"  Conditioned dims per star: "
          f"min={n_cond_per_star.min():.0f}, "
          f"max={n_cond_per_star.max():.0f}, "
          f"mean={n_cond_per_star.mean():.1f}")
    print(f"  Free dims per star (to be inferred): "
          f"{NUM_NODES - n_cond_per_star.mean():.1f} on average")

    # ---- Sample posteriors ----
    print(f"\n--- Sampling ({args.num_samples} draws/star, {args.steps} steps) ---")
    t0 = time.time()

    all_samples = sample_posterior(
        model=model,
        condition_values=condition_values,
        condition_mask=condition_mask,
        observed_mask=observed_mask,
        errors=errors,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        steps=args.steps,
        device=device,
    )

    elapsed = time.time() - t0
    print(f"\n  Sampling completed in {elapsed:.1f}s "
          f"({elapsed / len(obs_df):.2f}s/star)")

    # ---- Save results ----
    output_path = args.output or os.path.join(args.model_dir, "posteriors.parquet")

    # Use an identifier column if available, otherwise use row index
    star_ids = obs_df.index.values if obs_df.index.name else np.arange(len(obs_df))

    print(f"\n--- Saving results ---")
    result_df = samples_to_dataframe(all_samples, norm_stats, star_ids=star_ids)
    result_df.to_parquet(output_path, index=False)
    print(f"  Posteriors saved to {output_path}")
    print(f"  Shape: {result_df.shape[0]:,} rows "
          f"({len(obs_df)} stars x {args.num_samples} samples)")

    # ---- Summary statistics ----
    print(f"\n--- Summary ---")
    print(f"  Stars:      {len(obs_df)}")
    print(f"  Samples:    {args.num_samples}/star")
    print(f"  ODE steps:  {args.steps}")
    print(f"  Total time: {elapsed:.1f}s")

    # Print mean/std of inferred intrinsic params for the first star
    if len(obs_df) > 0:
        first_star = all_samples[0]  # (num_samples, NUM_NODES)
        first_star_phys = norm_stats.denormalize(first_star)
        print(f"\n  Posterior summary for first star:")
        for i, col in enumerate(INTRINSIC_COLS):
            vals = first_star_phys[:, i].numpy()
            print(f"    {col:>12s}: {vals.mean():10.4f} +/- {vals.std():.4f}")


if __name__ == "__main__":
    main()
