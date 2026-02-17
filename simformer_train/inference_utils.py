"""
Normalization utilities for SimFormer inference.

Loads the normalization statistics saved during training (norm_stats.npz)
and provides normalize/denormalize transforms for both torch tensors and
numpy arrays.
"""

import numpy as np
import torch
from typing import Optional, Sequence, Union


class NormStats:
    """Load and apply normalization statistics saved during training.

    The training script (`train_mock_galaxy.py`) saves a ``norm_stats.npz``
    file containing per-column means, standard deviations, and column names.
    This class loads those stats and provides convenience methods for
    normalizing raw observations before sampling and denormalizing model
    outputs back to physical units.

    Usage::

        stats = NormStats("output/norm_stats.npz")

        # Normalize observed values before passing to the model
        x_norm = stats.normalize(x_raw)

        # Denormalize model output to physical units
        x_phys = stats.denormalize(samples)

        # Column name lookup
        idx = stats.column_index("logAge")
    """

    def __init__(self, norm_stats_path: str):
        """Load normalization statistics from an .npz file.

        Args:
            norm_stats_path: Path to the ``norm_stats.npz`` file produced
                by ``train_mock_galaxy.py``.
        """
        data = np.load(norm_stats_path, allow_pickle=True)
        self.means = data["means"].astype(np.float32)       # (num_nodes,)
        self.stds = data["stds"].astype(np.float32)         # (num_nodes,)
        self.columns = list(data["columns"])                 # list[str]
        self.num_nodes = len(self.means)

        # Column name → index mapping
        self._col_to_idx = {name: i for i, name in enumerate(self.columns)}

    # ------------------------------------------------------------------
    # Column lookup
    # ------------------------------------------------------------------

    def column_index(self, name: str) -> int:
        """Return the integer index for a column name.

        Raises KeyError if the name is not found.
        """
        return self._col_to_idx[name]

    def column_indices(self, names: Sequence[str]) -> list:
        """Return a list of integer indices for the given column names."""
        return [self._col_to_idx[n] for n in names]

    # ------------------------------------------------------------------
    # Torch operations
    # ------------------------------------------------------------------

    def _get_torch_stats(
        self,
        column_indices: Optional[Sequence[int]],
        device: Union[str, torch.device] = "cpu",
    ):
        """Return (means, stds) as torch tensors, optionally sub-selected."""
        if column_indices is not None:
            m = torch.tensor(self.means[list(column_indices)], device=device)
            s = torch.tensor(self.stds[list(column_indices)], device=device)
        else:
            m = torch.tensor(self.means, device=device)
            s = torch.tensor(self.stds, device=device)
        return m, s

    def normalize(
        self,
        values: torch.Tensor,
        column_indices: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        """Normalize values: ``(x - mean) / std``.

        Args:
            values: Tensor of shape ``(..., D)`` where ``D`` is either
                ``num_nodes`` (when *column_indices* is None) or
                ``len(column_indices)``.
            column_indices: Optional subset of column indices.

        Returns:
            Normalized tensor, same shape as input.
        """
        m, s = self._get_torch_stats(column_indices, device=values.device)
        return (values - m) / s

    def denormalize(
        self,
        values: torch.Tensor,
        column_indices: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        """Denormalize values: ``x * std + mean``.

        Args:
            values: Tensor of shape ``(..., D)`` in normalized space.
            column_indices: Optional subset of column indices.

        Returns:
            Denormalized tensor in physical units.
        """
        m, s = self._get_torch_stats(column_indices, device=values.device)
        return values * s + m

    # ------------------------------------------------------------------
    # Numpy operations
    # ------------------------------------------------------------------

    def normalize_numpy(
        self,
        values: np.ndarray,
        column_indices: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """Normalize values: ``(x - mean) / std`` (numpy version).

        Args:
            values: Array of shape ``(..., D)``.
            column_indices: Optional subset of column indices.

        Returns:
            Normalized array.
        """
        if column_indices is not None:
            m = self.means[list(column_indices)]
            s = self.stds[list(column_indices)]
        else:
            m, s = self.means, self.stds
        return (values - m) / s

    def denormalize_numpy(
        self,
        values: np.ndarray,
        column_indices: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """Denormalize values: ``x * std + mean`` (numpy version).

        Args:
            values: Array of shape ``(..., D)`` in normalized space.
            column_indices: Optional subset of column indices.

        Returns:
            Denormalized array in physical units.
        """
        if column_indices is not None:
            m = self.means[list(column_indices)]
            s = self.stds[list(column_indices)]
        else:
            m, s = self.means, self.stds
        return values * s + m
