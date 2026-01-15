import torch
import torch.optim as optim
import wandb
import copy
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
        condition_mask_generator=None,
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
            condition_mask_generator=condition_mask_generator,
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
            device=device,
        )

        # Physics-specific parameters
        self.physics_residual_fn = physics_residual_fn
        self.physics_feature_indices = physics_feature_indices
        self.unrolling_steps = int(unrolling_steps)
        self.residual_weight_power = float(residual_weight_power)
        self.physics_loss_weight = float(physics_loss_weight)
        self.use_conflict_free = bool(use_conflict_free)

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
        t_scalar = float(t[0, 0, 0].item())
        cond3 = condition_mask.to(xt.dtype, copy=False).unsqueeze(-1)  # (B,N,1)
        free3 = 1.0 - cond3

        if self.unrolling_steps == 1:
            # Single step prediction using the velocity field
            v = self.model(t, xt, node_ids, cond3, edge_mask, x_errors)  # (B,N,1)
            v = v * free3                                               # project
            # We integrate from t to 1 to get x1
            x1_pred = xt + (1.0 - t_scalar) * v
            # Clamp state on fixed (conditioned on) coords
            x1_pred = x1_pred * free3 + x_fixed.unsqueeze(-1) * cond3   # clamp
            return x1_pred.squeeze(-1)                                  # (B,N)

        # Multiple unrolling steps for better accuracy
        dt = (1.0 - t_scalar) / self.unrolling_steps
        x_cur = xt.clone()
        cur_t = t_scalar

        for _ in range(self.unrolling_steps):
            # Create time tensor for current step
            t_cur = torch.full((batch_size, 1, 1), cur_t, device=self.device, dtype=xt.dtype)
            # predict velocity
            v = self.model(t_cur, x_cur, node_ids, cond3, edge_mask, x_errors)  # (B,N,1)
            v = v * free3
            # Euler step
            x_cur = x_cur + dt * v
            # Clamp state to conditional manifold after each step
            x_cur = x_cur * free3 + x_fixed.unsqueeze(-1) * cond3               # clamp
            cur_t = min(cur_t + dt, 1.0)

        return x_cur.squeeze(-1)

    def physics_residual_loss(self, x1_pred, t_scalar, condition_mask, x_errors=None):
        """
        Compute physics residual loss for predicted final state.

        Args:
            x1_pred: Predicted final state [B, num_nodes]
            t_scalar: Scalar time value for weighting
            x_errors: Optional error tensor [B, num_nodes]

        Returns:
            Weighted physics residual loss
        """
        device = getattr(self, "device", x1_pred.device)
        if self.physics_residual_fn is None:
            return torch.tensor(0.0, device=device, dtype=x1_pred.dtype)

        if not torch.is_tensor(t_scalar):
            t_scalar = torch.tensor(t_scalar, device=device, dtype=x1_pred.dtype)

        # New residual API: (x, condition_mask, x_errors)
        try:
            residual = self.physics_residual_fn(x1_pred, condition_mask, x_errors)
        except TypeError:
            try:
                residual = self.physics_residual_fn(x1_pred, condition_mask)
            except TypeError:
                if getattr(self, "physics_feature_indices", None) is not None:
                    x_phys = x1_pred[:, self.physics_feature_indices]
                    x_err_phys = x_errors[:, self.physics_feature_indices] if x_errors is not None else None
                    try:
                        residual = self.physics_residual_fn(x_phys, x_err_phys)
                    except TypeError:
                        residual = self.physics_residual_fn(x_phys)
                else:
                    try:
                        residual = self.physics_residual_fn(x1_pred, x_errors)
                    except TypeError:
                        residual = self.physics_residual_fn(x1_pred)

        weight = (t_scalar ** self.residual_weight_power) * self.physics_loss_weight

        if residual.dim() == 0:
            return weight * residual
        elif residual.dim() == 1:
            return weight * residual.mean()
        else:
            return weight * residual.float().pow(2).mean()


    def training_step(self, x0, x1, node_ids, condition_mask, edge_mask, x_errors):
        """
        Perform one training step with optional conflict-free gradient updates.
        NOW INCLUDES ERRORS.
        """
        B, N = x0.shape
        # time draw
        t = self.sample_t(B).view(B, 1, 1)            # (B,1,1), same dtype as x0 later
        t_scalar = float(t[0, 0, 0].item())
        k = (1.0 - self.sigma_min)

        # Interpolate + clamp (grad-friendly)
        xt = (1 - k * t) * x0.unsqueeze(-1) + t * x1.unsqueeze(-1)       # (B,N,1)
        cond3 = condition_mask.to(x0.dtype, copy=False).unsqueeze(-1)    # (B,N,1)
        free3 = 1.0 - cond3
        xt = xt * free3 + x1.unsqueeze(-1).detach() * cond3  # clamp state on cond dims

        # --- Compute losses ---
        loss_fm = self.flow_matching_loss(x0, x1, node_ids, condition_mask, edge_mask, x_errors)

        if self.physics_residual_fn is not None:
            # Predict x1 via unrolling (internally projects & clamps each step)
            x1_pred = self.predict_x1_with_unrolling(xt, t, node_ids, condition_mask, edge_mask, x_errors, x1)
            loss_physics = self.physics_residual_loss(x1_pred, t_scalar, condition_mask, x_errors)
        else:
            loss_physics = torch.tensor(0.0, device=self.device, dtype=x0.dtype)

        total_loss = loss_fm + loss_physics

        # --- Optim step ---
        self.optimizer.zero_grad(set_to_none=True)

        if self.use_conflict_free and (self.physics_residual_fn is not None):
            # Compute param-wise grads separately
            grads_fm = torch.autograd.grad(
                loss_fm, self.model.parameters(), retain_graph=True, allow_unused=True
            )
            grads_r = torch.autograd.grad(
                loss_physics, self.model.parameters(), retain_graph=False, allow_unused=True
            )

            # ConFIG combine
            grad_updates = self.config_optimizer.compute_conflict_free_update(list(grads_fm), list(grads_r))

            # Load combined grads into .grad and step
            for p, g in zip(self.model.parameters(), grad_updates):
                if g is not None:
                    p.grad = g
            self.optimizer.step()
        else:
            # Single objective (or no physics): standard backward
            total_loss.backward()
            self.optimizer.step()

        return {
            "loss": float(total_loss.detach()),
            "loss_fm": float(loss_fm.detach()),
            "loss_physics": float(loss_physics.detach()),
        }

    @torch.no_grad()
    def validate(self, num_batches=50):
        if self.val_data is None:
            return None, None, None

        self.model.eval()
        total_val, total_fm, total_phys = 0.0, 0.0, 0.0

        for _ in range(num_batches):
            x1, x_errors = self.sample_batch(from_val=True)
            x0 = torch.randn_like(x1)
            condition_mask = next(self.condition_mask_generator).to(self.device, dtype=torch.float32)
            edge_mask = self.build_edge_mask(dense_ratio=0.6)

            loss_fm = self.flow_matching_loss(x0, x1, self.node_ids, condition_mask, edge_mask, x_errors)
            total_fm += float(loss_fm)

            if self.physics_residual_fn is not None:
                B = x0.shape[0]
                t = self.sample_t(B).view(B, 1, 1)
                k = (1.0 - self.sigma_min)
                xt = (1 - k * t) * x0.unsqueeze(-1) + t * x1.unsqueeze(-1)
                cond3 = condition_mask.unsqueeze(-1)
                xt = xt * (1.0 - cond3) + x1.unsqueeze(-1) * cond3  # deterministic clamp (no detach ok in no_grad)

                x1_pred = self.predict_x1_with_unrolling(xt, t, self.node_ids, condition_mask, edge_mask, x_errors, x1)
                loss_phys = self.physics_residual_loss(x1_pred, float(t[0, 0, 0].item()), condition_mask, x_errors)
                total_phys += float(loss_phys)
                total_val += float(loss_fm + loss_phys)
            else:
                total_val += float(loss_fm)

        self.model.train()
        return total_val / num_batches, total_fm / num_batches, (total_phys / num_batches if self.physics_residual_fn is not None else 0.0)

    # ---------- Fit (unchanged, except uses validate above) ----------
    def fit(self, epochs=100, verbose=True):
        best_loss, no_improve, global_step = float("inf"), 0, 0

        for epoch in range(epochs):
            self.model.train()
            meter = {"loss": 0.0, "loss_fm": 0.0, "loss_physics": 0.0}

            for step in range(self.inner_train_loop_size):
                x1, x_errors = self.sample_batch(from_val=False)
                x0 = torch.randn_like(x1)

                condition_mask = next(self.condition_mask_generator).to(self.device, dtype=torch.float32)
                edge_mask = self.build_edge_mask(dense_ratio=0.6)

                out = self.training_step(x0, x1, self.node_ids, condition_mask, edge_mask, x_errors)
                for k in meter:
                    meter[k] += out[k] / self.inner_train_loop_size

                global_step += 1
                if self.use_wandb and (step % 100 == 0):
                    wandb.log({
                        "train_loss_step": out["loss"],
                        "train_loss_fm_step": out["loss_fm"],
                        "train_loss_physics_step": out["loss_physics"],
                        "epoch": epoch + 1,
                        "step": global_step,
                    })

            val_loss, val_fm, val_phys = (self.validate() if (epoch % self.val_every_n_epochs == 0) else (None, None, None))

            if verbose:
                msg = f"Epoch {epoch+1}: FM={meter['loss_fm']:.6f}, Phys={meter['loss_physics']:.6f}, Total={meter['loss']:.6f}"
                if val_loss is not None:
                    msg += f"\n           Val: FM={val_fm:.6f}, Phys={val_phys:.6f}, Total={val_loss:.6f}"
                print(msg)

            if self.use_wandb:
                log = {"train_loss_epoch": meter["loss"], "train_loss_fm_epoch": meter["loss_fm"],
                       "train_loss_physics_epoch": meter["loss_physics"], "epoch": epoch + 1}
                if val_loss is not None:
                    log.update({"val_loss": val_loss, "val_loss_fm": val_fm, "val_loss_physics": val_phys})
                wandb.log(log)

            monitor = val_loss if val_loss is not None else meter["loss"]
            if monitor < best_loss:
                best_loss, no_improve = monitor, 0
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                if self.use_wandb:
                    wandb.run.summary["best_loss"] = best_loss
                    wandb.run.summary["best_epoch"] = epoch + 1
            else:
                no_improve += 1
                if no_improve >= self.early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}.")
                    break

        if self.best_model_state is not None:
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