import torch
from typing import List


class ConFIGOptimizer:
    """Conflict-Free Gradient optimizer for multi-objective optimization."""

    @staticmethod
    def orthogonality_operator(g1: torch.Tensor, g2: torch.Tensor) -> torch.Tensor:
        """Compute orthogonality operator O(g1, g2) = g2 - (g1·g2/|g1|²)g1"""
        # Flatten gradients
        g1_flat = g1.flatten()
        g2_flat = g2.flatten()

        dot_product = (g1_flat * g2_flat).sum()
        g1_norm_sq = (g1_flat * g1_flat).sum()

        if g1_norm_sq < 1e-8:
            return g2

        return g2 - (dot_product / g1_norm_sq) * g1

    @staticmethod
    def unit_vector(g: torch.Tensor) -> torch.Tensor:
        """Compute unit vector U(g) = g/|g|"""
        g_flat = g.flatten()
        norm = torch.norm(g_flat, p=2)

        if norm < 1e-8:
            return g

        return g / norm

    @staticmethod
    def compute_conflict_free_update(grads_fm: List[torch.Tensor], grads_r: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Compute conflict-free gradient updates for all parameters.

        Args:
            grads_fm: List of gradients from flow matching loss
            grads_r: List of gradients from residual loss

        Returns:
            List of conflict-free gradient updates
        """
        updates = []

        for g_fm, g_r in zip(grads_fm, grads_r):
            if g_fm is None or g_r is None:
                updates.append(g_fm if g_fm is not None else g_r)
                continue

            # Flatten for computation
            g_fm_flat = g_fm.flatten()
            g_r_flat = g_r.flatten()

            # Check for zero gradients
            if torch.norm(g_fm_flat) < 1e-8 or torch.norm(g_r_flat) < 1e-8:
                updates.append(g_fm + g_r)
                continue

            # Compute orthogonality operations
            o_fm_r = ConFIGOptimizer.orthogonality_operator(g_fm, g_r)
            o_r_fm = ConFIGOptimizer.orthogonality_operator(g_r, g_fm)

            # Compute unit vectors and combine
            u_fm_r = ConFIGOptimizer.unit_vector(o_fm_r)
            u_r_fm = ConFIGOptimizer.unit_vector(o_r_fm)

            # Compute gv
            gv = ConFIGOptimizer.unit_vector(u_fm_r + u_r_fm)
            gv_flat = gv.flatten()

            # Compute final update
            g_update_flat = ((g_fm_flat @ gv_flat) + (g_r_flat @ gv_flat)) * gv_flat
            g_update = g_update_flat.reshape(g_fm.shape)

            updates.append(g_update)

        return updates
