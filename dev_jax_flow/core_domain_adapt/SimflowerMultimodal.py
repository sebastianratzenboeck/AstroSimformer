import jax
import jax.numpy as jnp
from typing import Dict, Any
from multimodal_domain_adapter import ModalityDomainAdapter
from flow_loss_forward import SimformerLossWrapper


class SimformerMultimodalFlowModel:
    def __init__(self, domain_adapter: ModalityDomainAdapter, loss_wrapper: SimformerLossWrapper, config: Dict[str, Any]):
        self.domain_adapter = domain_adapter
        self.loss_wrapper = loss_wrapper  # SimformerLossWrapper instance
        self.config = config      # includes t0, t1, etc.

    def compute_total_loss(self, params, batch, key):
        # Unpack
        x_s, theta_s = batch['source']
        x_t = batch.get('target', None)
        x_s_pair = batch.get('x_s_pair', None)
        x_t_pair = batch.get('x_t_pair', None)

        # Domain adapter: returns joint embedding and loss dict
        z_joint, loss_dict = self.domain_adapter.forward(x_s, x_t, x_s_pair, x_t_pair, rng=key)
        # Reshape to (B, T, 1) to be processed correctly in the transformer & flow model
        z_input = z_joint[:, :, None] if z_joint.ndim == 2 else z_joint
        theta_input = theta_s[:, :, None] if theta_s.ndim == 2 else theta_s
        # Join with theta
        z_merged = jnp.concatenate([z_input, theta_input], axis=1)

        # Flow matching loss (includes masking, noise, edge logic)
        fm_loss = self.loss_wrapper.forward(params, z_merged, key)

        # Aggregate all auxiliary losses
        scaled_losses = {}
        aux_total = 0.0
        for name, value in loss_dict.items():
            weight = self.config.get(f'lambda_{name}', 1.0)
            scaled_value = weight * value
            scaled_losses[name] = scaled_value
            aux_total += scaled_value

        aux_weight = self.config.get("lambda_aux", 1.0)
        total_loss = fm_loss + aux_weight * aux_total

        return total_loss, {'flow_matching': fm_loss, **scaled_losses, 'total_loss': total_loss}

    def train_step(self, state, batch, key):
        def loss_fn_wrapper(params):
            loss, logs = self.compute_total_loss(params, batch, key)
            return loss, logs

        grad_fn = jax.value_and_grad(loss_fn_wrapper, has_aux=True)
        (loss, logs), grads = grad_fn(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state, logs

