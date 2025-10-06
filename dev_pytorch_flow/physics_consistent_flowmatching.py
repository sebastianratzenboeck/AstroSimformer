import torch
import torch.optim as optim
from typing import Callable, Optional, List, Dict
from conflict_free_grad import ConFIGOptimizer

C_pc2cm = 3.085677581491367e+18

class PhysicsConsistentFlowMatchingTrainer:
    """
    Enhanced version of FlowMatchingTrainer that incorporates physics constraints
    using conflict-free gradient updates.
    """
    def __init__(
        self,
        model,
        data,
        data_errors=None,  # NEW: errors tensor with same shape as data
        # New
        physics_residual_fn: Optional[Callable] = None,
        physics_feature_indices: Optional[List[int]] = None,
        # ---
        batch_size=128,
        lr=1e-3,
        sigma_min=0.001,
        time_prior_exponent=1.0,
        inner_train_loop_size=1000,
        early_stopping_patience=20,
        # New
        unrolling_steps=1,          # Number of unrolling steps for better x1 prediction
        residual_weight_power=1.0,  # Power for time-based residual weighting
        use_conflict_free=True,     # Whether to use conflict-free gradients
        # ---
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.data = torch.tensor(data, dtype=torch.float32).to(device)

        # NEW: Store errors if provided
        if data_errors is not None:
            self.data_errors = torch.tensor(data_errors, dtype=torch.float32).to(device)
        else:
            # If no errors provided, use ones (no normalization effect)
            self.data_errors = torch.ones_like(self.data)

        self.batch_size = batch_size
        self.lr = lr
        self.sigma_min = sigma_min
        self.time_prior_exponent = time_prior_exponent
        self.inner_train_loop_size = inner_train_loop_size
        self.early_stopping_patience = early_stopping_patience
        self.device = device

        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        # New
        self.config_optimizer = ConFIGOptimizer() if use_conflict_free else None
        # ---
        self.best_model_state = None

        self.num_nodes = self.data.shape[1]
        self.node_ids = torch.arange(self.num_nodes).unsqueeze(0).repeat(self.batch_size, 1).to(device)

        # Physics-specific parameters (new)
        self.physics_residual_fn = physics_residual_fn
        self.physics_feature_indices = physics_feature_indices
        self.unrolling_steps = unrolling_steps
        self.residual_weight_power = residual_weight_power
        self.use_conflict_free = use_conflict_free

    def sample_batch(self):
        """Sample batch of data and corresponding errors."""
        idx = torch.randint(0, self.data.shape[0], (self.batch_size,))
        return self.data[idx], self.data_errors[idx]

    def build_condition_mask(self, prob=0.333):
        mask = torch.bernoulli(torch.full((self.batch_size, self.num_nodes), prob)).bool().to(self.device)
        all_true = mask.all(dim=1, keepdim=True)
        mask = mask & ~all_true
        return mask.unsqueeze(-1).float()

    def build_edge_mask(self, dense_ratio=0.7):
        n_dense = int(self.batch_size * dense_ratio)
        dense = torch.ones((n_dense, self.num_nodes, self.num_nodes), dtype=torch.bool)
        sparse = []
        for _ in range(self.batch_size - n_dense):
            m = torch.ones((self.num_nodes, self.num_nodes), dtype=torch.bool)
            drop = torch.rand(self.num_nodes) < 0.2
            m[drop] = False
            m[:, drop] = False
            m[drop, drop] = True
            sparse.append(m)
        if sparse:
            sparse = torch.stack(sparse, dim=0)
            masks = torch.cat([dense, sparse], dim=0)
        else:
            masks = dense
        indices = torch.randint(0, masks.shape[0], (self.batch_size,))
        return masks[indices].to(self.device)

    def sample_t(self, batch_size):
        t = torch.rand(batch_size, device=self.device)
        return t.pow(1 / (1 + self.time_prior_exponent))

    # New
    def predict_x1_with_unrolling(self, xt, t, node_ids, condition_mask, edge_mask, x_errors, x_fixed):
        """
        Predict final state x1 using unrolling for better accuracy.

        NOW INCLUDES ERRORS as input to the model.

        Args:
            xt: Current interpolated state
            t: Current time (shape: [B, 1, 1])
            node_ids: Node IDs
            condition_mask: Condition mask
            edge_mask: Edge mask
            x_errors: Measurement errors [B, num_nodes]

        Returns:
            Predicted x1 state
        """
        batch_size = xt.shape[0]
        t_scalar = t[0, 0, 0].item()  # Get scalar time value

        if self.unrolling_steps == 1:
            # Single step prediction using the velocity field
            v = self.model(t, xt, node_ids, condition_mask, edge_mask, x_errors)  # ADDED x_errors
            v = v * (1.0 - condition_mask)  # project velocity
            # We integrate from t to 1 to get x1
            x1_pred = xt + (1 - t_scalar) * v
            # Clamp state on fixed (conditioned on) coords
            x1_pred = x1_pred * (1.0 - condition_mask) + x_fixed.unsqueeze(-1) * condition_mask
            return x1_pred.squeeze(-1)

        # Multiple unrolling steps for better accuracy
        dt = (1 - t_scalar) / self.unrolling_steps
        x_current = xt.clone()
        current_t = t_scalar

        for step in range(self.unrolling_steps):
            # Create time tensor for current step
            t_current = torch.full((batch_size, 1, 1), current_t, device=self.device)

            # Predict velocity
            # if step == 0:
            #     # Use gradient-tracked computation for first step
            #     velocity = self.model(t_current, x_current, node_ids, condition_mask, edge_mask,
            #                           x_errors)  # ADDED x_errors
            # else:
            #     # No gradient tracking for subsequent steps (saves memory)
            #     with torch.no_grad():
            #         velocity = self.model(t_current, x_current, node_ids, condition_mask, edge_mask,
            #                               x_errors)  # ADDED x_errors
            # x_current = x_current + dt * velocity  # Euler integration step

            # predict velocity
            v = self.model(t_current, x_current, node_ids, condition_mask, edge_mask, x_errors)
            v = v * (1.0 - condition_mask)  # project to free dims
            # Euler step
            x_current = x_current + dt * v
            # Clamp state to conditional manifold after each step
            x_current = x_current * (1.0 - condition_mask) + x_fixed.unsqueeze(-1) * condition_mask
            current_t = min(current_t + dt, 1.0)

        return x_current.squeeze(-1)

    def flow_matching_loss(self, x0, x1, t, xt, node_ids, condition_mask, edge_mask, x_errors):
        """Compute standard flow matching loss. NOW INCLUDES ERRORS."""
        pred_velocity = self.model(t, xt, node_ids, condition_mask, edge_mask, x_errors)  # (B,M,1)
        # Target FM velocity (same shape after unsqueeze)
        velocity = (x1 - (1 - self.sigma_min) * x0).unsqueeze(-1)
        # project both to the free subspace
        free = (1.0 - condition_mask)
        pred_velocity = pred_velocity * free
        diff = (pred_velocity - velocity).squeeze(-1) ** 2

        # diff = (pred_velocity.squeeze(-1) - velocity) ** 2
        # diff = torch.where(loss_mask.squeeze(-1), 0.0, diff)

        loss_mask = condition_mask.bool()
        num_elements = torch.sum(~loss_mask, axis=-2, keepdim=True)
        diff = torch.where(num_elements > 0, diff / num_elements, 0.0)
        # Mean over batch
        loss = torch.mean(diff)
        return loss

    def physics_residual_loss(self, x1_pred, t_scalar, x_errors=None):
        """
        Compute physics residual loss for predicted final state.

        Args:
            x1_pred: Predicted final state [B, num_nodes]
            t_scalar: Scalar time value for weighting
            x_errors: Optional error tensor [B, num_nodes]

        Returns:
            Weighted physics residual loss
        """
        if self.physics_residual_fn is None:
            return torch.tensor(0.0, device=self.device)

        # Extract relevant features if specified
        if self.physics_feature_indices is not None:
            x_physics = x1_pred[:, self.physics_feature_indices]
            if x_errors is not None:
                x_errors_physics = x_errors[:, self.physics_feature_indices]
            else:
                x_errors_physics = None
        else:
            x_physics = x1_pred
            x_errors_physics = x_errors

        # Compute residual (pass errors if residual function supports it)
        try:
            residual = self.physics_residual_fn(x_physics, x_errors_physics)
        except TypeError:
            # Fallback for residual functions that don't accept errors
            residual = self.physics_residual_fn(x_physics)

        # Weight by t^p (emphasize later times when predictions are more accurate)
        weight = t_scalar ** self.residual_weight_power

        # Compute MSE of residual
        loss = weight * torch.mean(residual ** 2)

        return loss

    def training_step(self, x0, x1, node_ids, condition_mask, edge_mask, x_errors):
        """
        Perform one training step with optional conflict-free gradient updates.
        NOW INCLUDES ERRORS.
        """
        B, N = x0.shape
        t = self.sample_t(B).reshape(B, 1, 1)
        t_scalar = t[0, 0, 0].item()

        # Compute interpolated state
        xt = (1 - (1 - self.sigma_min) * t) * x0.unsqueeze(-1) + t * x1.unsqueeze(-1)
        xt = torch.where(condition_mask.bool(), x1.unsqueeze(-1), xt)

        if (self.physics_residual_fn is not None) and self.use_conflict_free:
            # Compute losses separately for conflict-free updates

            # Flow matching loss
            loss_fm = self.flow_matching_loss(x0, x1, t, xt, node_ids, condition_mask, edge_mask, x_errors)

            # Physics residual loss with unrolling
            x1_pred = self.predict_x1_with_unrolling(xt, t, node_ids, condition_mask, edge_mask, x_errors, x1)
            loss_physics = self.physics_residual_loss(x1_pred, t_scalar)

            # Compute gradients separately
            self.optimizer.zero_grad()

            # Get gradients for flow matching loss
            grads_fm = torch.autograd.grad(
                loss_fm,
                self.model.parameters(),
                retain_graph=True,
                create_graph=False
            )

            # Get gradients for physics loss
            if loss_physics.item() > 0:
                grads_r = torch.autograd.grad(
                    loss_physics,
                    self.model.parameters(),
                    retain_graph=False,
                    create_graph=False
                )

                # Compute conflict-free updates
                grad_updates = self.config_optimizer.compute_conflict_free_update(
                    list(grads_fm), list(grads_r)
                )

                # Apply conflict-free gradients
                for param, grad_update in zip(self.model.parameters(), grad_updates):
                    param.grad = grad_update

            else:
                # Only flow matching gradients
                for param, grad in zip(self.model.parameters(), grads_fm):
                    param.grad = grad

            total_loss = loss_fm + loss_physics

        else:
            # Standard training (weighted sum of losses)
            loss_fm = self.flow_matching_loss(x0, x1, t, xt, node_ids, condition_mask, edge_mask, x_errors)

            if self.physics_residual_fn is not None:
                x1_pred = self.predict_x1_with_unrolling(xt, t, node_ids, condition_mask, edge_mask, x_errors)
                loss_physics = self.physics_residual_loss(x1_pred, t_scalar)
                total_loss = loss_fm + loss_physics
            else:
                loss_physics = torch.tensor(0.0, device=self.device)
                total_loss = loss_fm

            self.optimizer.zero_grad()
            total_loss.backward()

        self.optimizer.step()

        return {
            'loss_fm': loss_fm.item(),
            'loss_physics': loss_physics.item(),
            'loss_total': total_loss.item()
        }

    def fit(self, epochs=100, verbose=True):
        """Train the model with physics constraints."""
        best_loss = float("inf")
        no_improve = 0

        for epoch in range(epochs):
            metrics = {'loss_fm': 0.0, 'loss_physics': 0.0, 'loss_total': 0.0}

            for _ in range(self.inner_train_loop_size):
                x1, x_errors = self.sample_batch()  # NOW RETURNS ERRORS TOO
                x0 = torch.randn_like(x1)

                condition_mask = self.build_condition_mask()
                edge_mask = self.build_edge_mask(dense_ratio=0.6)

                step_metrics = self.training_step(
                    x0, x1, self.node_ids, condition_mask, edge_mask, x_errors  # PASS ERRORS
                )

                for key in metrics:
                    metrics[key] += step_metrics[key] / self.inner_train_loop_size

            if verbose:
                print(f"Epoch {epoch + 1}: FM Loss = {metrics['loss_fm']:.6f}, "
                      f"Physics Loss = {metrics['loss_physics']:.6f}, "
                      f"Total = {metrics['loss_total']:.6f}")

            if metrics['loss_total'] < best_loss:
                best_loss = metrics['loss_total']
                no_improve = 0
                self.best_model_state = self.model.state_dict()
            else:
                no_improve += 1

            if no_improve >= self.early_stopping_patience:
                if verbose:
                    print("Early stopping triggered.")
                break

        self.model.load_state_dict(self.best_model_state)
        return self.model


def create_photometric_residual(
        mu_sed: torch.Tensor,
        phi_sed: torch.Tensor,
        k_lambda: torch.Tensor,
        lam_x_dw_x_filter: torch.Tensor,
        means: torch.Tensor,
        stds: torch.Tensor,
        zero_pts_phot: torch.Tensor,
        coeff_indices: List[int],
        log_scale_idx: int,
        av_idx: int,
        distance_idx: int,
        photometry_indices: List[int],
        photometry_error_indices: List[int],  # Indices in the error vector
        normalize_by_errors: bool = True,
        handle_nans: bool = True
):
    """
    Create a residual function that enforces consistency between spectral parameters
    and observed photometry.

    This version assumes errors are passed separately to the model and stored in
    the trainer's data structure.

    Args:
        mu_sed (torch.Tensor): mean SED
        phi_sed (torch.Tensor): phi SED
        lam_x_dw_x_filter (torch.Tensor): product of d_log_lambda x wavelength grid x filter grid
        k_lambda (torch.Tensor): extrinction law lambda parameter
        zero_pts_phot (torch.Tensor): zero points of photometry
        coeff_indices: List of feature indices for spectral coefficients
        log_scale_idx: Feature index for log_scale parameter
        av_idx: Feature index for extinction (Av)
        distance_idx: Feature index for distance
        photometry_indices: List of feature indices for observed photometry
        photometry_error_indices: List of indices in the ERROR tensor for photometry errors
        normalize_by_errors: Whether to normalize residuals by measurement errors
        handle_nans: Whether to mask out NaN values in photometry (missing bands)

    Returns:
        Residual function that computes photometric consistency residual
    """
    b = torch.sum(lam_x_dw_x_filter, dim=1).unsqueeze(0)

    def residual_fn(x, x_errors=None):
        """
        Compute photometric consistency residual.

        Args:
            x: Predicted state with shape [batch_size, num_features]
            x_errors: Optional error tensor with shape [batch_size, num_features]

        Returns:
            Residual with shape [batch_size, n_bands]
        """
        # Extract spectral coefficients
        phot_i = x[:, photometry_indices] * stds[photometry_indices] + means[photometry_indices]
        log_scales_i = x[:, log_scale_idx] * stds[log_scale_idx] + means[log_scale_idx]
        coeffs_i = x[:, coeff_indices] * stds[coeff_indices] + means[coeff_indices]
        Av_i = x[:, av_idx] * stds[av_idx] + means[av_idx]
        dist_i = x[:, distance_idx] * stds[distance_idx] + means[distance_idx]

        f_r = (mu_sed.unsqueeze(0) + coeffs_i @ phi_sed.T) * torch.exp(log_scales_i.unsqueeze(-1)) / (dist_i.unsqueeze(-1) * C_pc2cm)
        f_ext_t = f_r * torch.exp(-k_lambda.unsqueeze(0) * Av_i.unsqueeze(-1))

        flux_in_filter_est = torch.sum(lam_x_dw_x_filter.unsqueeze(0) * f_ext_t.unsqueeze(0).transpose(0, 1), dim=-1) / b
        phot_est = -2.5 * torch.log10(flux_in_filter_est / zero_pts_phot.unsqueeze(0))

        # Compute residual (predicted - observed)
        residual = phot_est - phot_i

        # Handle NaN values (missing photometry)
        if handle_nans:
            residual = torch.where(torch.isnan(residual), torch.zeros_like(residual), residual)

        # Optionally normalize by measurement errors (chi-squared normalization)
        if normalize_by_errors and (x_errors is not None):
            # Extract errors for photometry bands
            phot_err = x_errors[:, photometry_error_indices]  # [B, n_bands]

            # Normalize: chi = (obs - model) / sigma
            residual = residual / (phot_err + 1e-8)

        return torch.mean(residual**2)

    return residual_fn