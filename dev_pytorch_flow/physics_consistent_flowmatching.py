import torch
import torch.optim as optim
import wandb
from typing import Callable, Optional, List, Dict
from simflower import FlowMatchingTrainer
from conflict_free_grad import ConFIGOptimizer

C_pc2cm = 3.085677581491367e+18

class PhysicsConsistentFlowMatchingTrainer(FlowMatchingTrainer):
    """
    Enhanced version of FlowMatchingTrainer that incorporates physics constraints
    using conflict-free gradient updates.
    """
    def __init__(
        self,
        model,
        data,
        data_errors=None,
        physics_residual_fn: Optional[Callable] = None,
        physics_feature_indices: Optional[List[int]] = None,
        batch_size=128,
        lr=1e-3,
        sigma_min=0.001,
        time_prior_exponent=1.0,
        inner_train_loop_size=1000,
        early_stopping_patience=20,
        val_split=0.15,
        val_every_n_epochs=1,
        unrolling_steps=1,
        residual_weight_power=1.0,
        physics_loss_weight=1.0,
        use_conflict_free=True,
        use_wandb=True,
        wandb_project="flow-matching-physics",
        wandb_config=None,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__(
            model=model,
            data=data,
            data_errors=data_errors,
            batch_size=batch_size,
            lr=lr,
            sigma_min=sigma_min,
            time_prior_exponent=time_prior_exponent,
            inner_train_loop_size=inner_train_loop_size,
            early_stopping_patience=early_stopping_patience,
            val_split=val_split,
            val_every_n_epochs=val_every_n_epochs,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            wandb_config=wandb_config,
            device=device
        )

        # Physics-specific parameters
        self.physics_residual_fn = physics_residual_fn
        self.physics_feature_indices = physics_feature_indices
        self.unrolling_steps = unrolling_steps
        self.residual_weight_power = residual_weight_power
        self.physics_loss_weight = physics_loss_weight
        self.use_conflict_free = use_conflict_free

        # Initialize conflict-free optimizer if needed
        self.config_optimizer = ConFIGOptimizer() if self.use_conflict_free else None

        # Add physics config to wandb
        if self.use_wandb:
            wandb.config.update({
                "physics_enabled": physics_residual_fn is not None,
                "unrolling_steps": unrolling_steps,
                "residual_weight_power": residual_weight_power,
                "use_conflict_free": self.use_conflict_free,
            })

    # New
    def predict_x1_with_unrolling(self, xt, t, node_ids, condition_mask, edge_mask, x_errors, x_fixed):
        """
        Predict final state x1 using unrolling for better accuracy.

        Args:
            xt: Current interpolated state
            t: Current time (shape: [B, 1, 1])
            node_ids: Node IDs
            condition_mask: Condition mask
            edge_mask: Edge mask
            x_errors: Measurement errors [B, num_nodes, 1]
            x_fixed: Fixed values for conditioned variables [B, num_nodes]

        Returns:
            Predicted x1 state [B, num_nodes]
        """
        batch_size = xt.shape[0]
        t_scalar = t[0, 0, 0].item()

        if self.unrolling_steps == 1:
            # Single step prediction using the velocity field
            v = self.model(t, xt, node_ids, condition_mask, edge_mask, x_errors)
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

        # Weight by t^p (emphasize later times when predictions are more accurate) x physics_loss_weight
        weight = t_scalar ** self.residual_weight_power * self.physics_loss_weight

        # Return scalar loss (already computed in residual_fn for photometry)
        if residual.dim() == 0:  # scalar
            return weight * residual
        else:
            return weight * torch.mean(residual ** 2)

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
            loss_fm = self.flow_matching_loss(x0, x1, node_ids, condition_mask, edge_mask, x_errors)

            # Physics residual loss with unrolling
            x1_pred = self.predict_x1_with_unrolling(xt, t, node_ids, condition_mask, edge_mask, x_errors, x1)
            loss_physics = self.physics_residual_loss(x1_pred, t_scalar, x_errors)

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
            loss_fm = self.flow_matching_loss(x0, x1, node_ids, condition_mask, edge_mask, x_errors)

            if self.physics_residual_fn is not None:
                x1_pred = self.predict_x1_with_unrolling(xt, t, node_ids, condition_mask, edge_mask, x_errors, x1)
                loss_physics = self.physics_residual_loss(x1_pred, t_scalar, x_errors)
                total_loss = loss_fm + loss_physics
            else:
                loss_physics = torch.tensor(0.0, device=self.device)
                total_loss = loss_fm

            self.optimizer.zero_grad()
            total_loss.backward()

        self.optimizer.step()

        return {
            'loss': total_loss.item(),
            'loss_fm': loss_fm.item(),
            'loss_physics': loss_physics.item()
        }

    @torch.no_grad()
    def validate(self, num_batches=50):
        """Compute validation loss with physics constraints."""
        if self.val_data is None:
            return None, None, None

        self.model.eval()
        total_val_loss = 0.0
        total_fm_loss = 0.0
        total_physics_loss = 0.0

        for _ in range(num_batches):
            x1, x_errors = self.sample_batch(from_val=True)
            x0 = torch.randn_like(x1)
            condition_mask = self.build_condition_mask()
            edge_mask = self.build_edge_mask(dense_ratio=0.6)

            # Flow matching loss
            loss_fm = self.flow_matching_loss(x0, x1, self.node_ids, condition_mask, edge_mask, x_errors)
            total_fm_loss += loss_fm.item()

            # Physics loss if enabled
            if self.physics_residual_fn is not None:
                B, N = x0.shape
                t = self.sample_t(B).reshape(B, 1, 1)
                t_scalar = t[0, 0, 0].item()
                xt = (1 - (1 - self.sigma_min) * t) * x0.unsqueeze(-1) + t * x1.unsqueeze(-1)
                xt = torch.where(condition_mask.bool(), x1.unsqueeze(-1), xt)

                x1_pred = self.predict_x1_with_unrolling(xt, t, self.node_ids, condition_mask, edge_mask, x_errors, x1)
                loss_physics = self.physics_residual_loss(x1_pred, t_scalar, x_errors)
                total_physics_loss += loss_physics.item()
                total_val_loss += loss_fm.item() + loss_physics.item()
            else:
                total_val_loss += loss_fm.item()

        self.model.train()

        avg_total = total_val_loss / num_batches
        avg_fm = total_fm_loss / num_batches
        avg_physics = total_physics_loss / num_batches if self.physics_residual_fn is not None else 0.0

        return avg_total, avg_fm, avg_physics

    def fit(self, epochs=100, verbose=True):
        """Train the model with physics constraints."""
        best_loss = float("inf")
        no_improve = 0
        global_step = 0

        for epoch in range(epochs):
            self.model.train()
            metrics = {'loss': 0.0, 'loss_fm': 0.0, 'loss_physics': 0.0}

            for step in range(self.inner_train_loop_size):
                x1, x_errors = self.sample_batch(from_val=False)
                x0 = torch.randn_like(x1)

                condition_mask = self.build_condition_mask()
                edge_mask = self.build_edge_mask(dense_ratio=0.6)

                step_metrics = self.training_step(x0, x1, self.node_ids, condition_mask, edge_mask, x_errors)

                for key in metrics:
                    metrics[key] += step_metrics[key] / self.inner_train_loop_size

                global_step += 1

                # Log to wandb every N steps
                if self.use_wandb and step % 100 == 0:
                    wandb.log({
                        "train_loss_step": step_metrics['loss'],
                        "train_loss_fm_step": step_metrics['loss_fm'],
                        "train_loss_physics_step": step_metrics['loss_physics'],
                        "epoch": epoch + 1,
                        "step": global_step
                    })

            # Validate periodically
            val_loss, val_fm, val_physics = None, None, None
            if epoch % self.val_every_n_epochs == 0:
                val_loss, val_fm, val_physics = self.validate()

            # Logging
            if verbose:
                log_str = f"Epoch {epoch + 1}: FM Loss = {metrics['loss_fm']:.6f}, Physics Loss = {metrics['loss_physics']:.6f}, Total = {metrics['loss']:.6f}"
                if val_loss is not None:
                    log_str += f"\n              Val: FM = {val_fm:.6f}, Physics = {val_physics:.6f}, Total = {val_loss:.6f}"
                print(log_str)

            if self.use_wandb:
                log_dict = {
                    "train_loss_epoch": metrics['loss'],
                    "train_loss_fm_epoch": metrics['loss_fm'],
                    "train_loss_physics_epoch": metrics['loss_physics'],
                    "epoch": epoch + 1
                }
                if val_loss is not None:
                    log_dict.update({
                        "val_loss": val_loss,
                        "val_loss_fm": val_fm,
                        "val_loss_physics": val_physics
                    })
                wandb.log(log_dict)

            # Early stopping based on validation loss (if available) or training loss
            monitor_loss = val_loss if val_loss is not None else metrics['loss']

            if monitor_loss < best_loss:
                best_loss = monitor_loss
                no_improve = 0
                self.best_model_state = self.model.state_dict()
                if self.use_wandb:
                    wandb.run.summary["best_loss"] = best_loss
                    wandb.run.summary["best_epoch"] = epoch + 1
            else:
                no_improve += 1

            if no_improve >= self.early_stopping_patience:
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch + 1}.")
                break

        # Load best model
        self.model.load_state_dict(self.best_model_state)

        if self.use_wandb:
            wandb.finish()

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