import jax
import jax.numpy as jnp
from typing import Dict, Callable, Optional, Any
from functools import partial


# === Vector field wrapper ===
class SimformerConditionalVelocity:
    def __init__(self, model_fn, params, condition_mask, condition_value, node_ids, edge_mask):
        self.model_fn = model_fn
        self.params = params
        self.condition_mask = condition_mask.astype(jnp.float32)
        self.condition_value = condition_value
        self.node_ids = node_ids
        self.edge_mask = edge_mask

    @partial(jax.jit, static_argnums=0)
    def __call__(self, t, x_flat, args):
        x = x_flat.reshape(self.node_ids.shape)
        B = x.shape[0]

        x = x * (1.0 - self.condition_mask) + self.condition_value * self.condition_mask
        t_batch = jnp.full((B, 1), t, dtype=x.dtype)

        v = self.model_fn(
            self.params,
            t_batch,
            x[..., None],
            self.node_ids,
            self.condition_mask,
            edge_mask=self.edge_mask
        )
        v = v.squeeze(-1)
        v = v * (1.0 - self.condition_mask)
        return v.reshape(-1)

def sample_batched_flow(
    model_fn: Callable,
    params,
    key: jnp.ndarray,                    # single PRNGKey
    condition_mask: jnp.ndarray,         # (B, M) or (B, M, D_mask)
    condition_values: jnp.ndarray,       # (B, M) or (B, M, D_val)
    node_ids: jnp.ndarray,               # (B, M)
    edge_masks: Optional[jnp.ndarray],   # (B, M, M)
    steps: int = 64,
    t0: float = 0.0,
    t1: float = 1.0,
) -> jnp.ndarray:
    """
    Draw B marginal samples over M nodes via simple Euler:
      - condition_masks may be (B,M) or (B,M,D); we collapse D→1 by OR.
      - condition_values may be (B,M) or (B,M,D); we take [:,:,0] as the clamp value.
      - After that x0 is shape (B,M), the integrator flattens it to (B*M,) internally.
    """
    x0 = jax.random.normal(key, node_ids.shape)
    dt = (t1 - t0) / steps
    ts = jnp.linspace(t0, t1, steps + 1)

    # — draw initial Gaussian for all B×M in one shot —
    # clamp any conditioned entries
    x0 = x0 * (1-condition_mask) + condition_values * condition_mask
    # x0 = jnp.where(condition_mask, condition_values, x0)
    x_flat0 = x0.reshape(-1)

    # build your velocity‐only wrapper
    vf = SimformerConditionalVelocity(
        model_fn, params,
        condition_mask,
        condition_values,
        node_ids,
        edge_masks
    )

    def step(x_flat, t):
        dx_flat = vf(t, x_flat, None)
        return x_flat + dt * dx_flat, None

    x_final_flat, _ = jax.lax.scan(step, x_flat0, ts[:-1])
    return x_final_flat.reshape(x0.shape)

# compile once into a single kernel
pure_batched_sampler = jax.jit(
    sample_batched_flow,
    static_argnames=("model_fn", "steps"),
)


def prepare_condition_vector(
    modality_inputs: Dict[str, Optional[jnp.ndarray]],
    domain_adapter,
    rng
):
    """
    Encodes available modality inputs using frozen domain adapter,
    returns stacked latent vector with NaNs for missing modalities.
    """
    assert isinstance(modality_inputs, dict)
    z_encoded = domain_adapter.encode(modality_inputs, rng=rng, is_training=False)
    z_joint = domain_adapter.merge_embeddings(z_encoded)  # (B, D)
    return z_joint


def sample_posterior(
    model_fn: Callable,
    params: Dict[str, Any],
    key: jax.random.PRNGKey,
    inputs: Dict[str, Optional[jnp.ndarray]],
    domain_adapter,
    theta_dim: int,
    num_samples: int = 1,
    t0: float = 0.0,
    t1: float = 1.0,
    steps: int = 64
):
    B = next(v for v in inputs.values() if v is not None).shape[0]  # N objects
    rng_encode, rng_sample = jax.random.split(key)

    # Encode modalities into latent z
    z_latent = prepare_condition_vector(inputs, domain_adapter, rng=rng_encode)  # (B, D_z)

    # Dummy theta tensor
    theta = jnp.zeros((B, theta_dim))

    # Combine z and theta (z = known inputs, theta = to be inferred)
    z_full = jnp.concatenate([z_latent, theta], axis=1)  # (B, D_total)
    D_z = z_latent.shape[1]
    D_total = z_full.shape[1]

    # Use all indices for output (including theta)
    node_ids = jnp.tile(jnp.arange(D_total), (B, 1))

    # Build condition mask and values
    condition_mask = jnp.zeros_like(z_full)
    condition_values = jnp.zeros_like(z_full)

    mask_z = ~jnp.isnan(z_latent)
    condition_mask = condition_mask.at[:, :D_z].set(mask_z.astype(jnp.float32))
    condition_values = condition_values.at[:, :D_z].set(jnp.nan_to_num(z_latent, nan=0.0))

    # Edge mask is False where values are NaN
    valid_mask = ~jnp.isnan(z_full)
    edge_mask = jnp.einsum("bi,bj->bij", valid_mask, valid_mask)
    edge_mask = jnp.repeat(edge_mask, num_samples, axis=0)

    # Repeat for sampling
    node_ids = jnp.repeat(node_ids, num_samples, axis=0)
    condition_mask = jnp.repeat(condition_mask, num_samples, axis=0)
    condition_values = jnp.repeat(condition_values, num_samples, axis=0)

    # Sample from the model
    post_samples = pure_batched_sampler(
        model_fn=model_fn,
        params=params,
        key=rng_sample,
        condition_mask=condition_mask,
        condition_values=condition_values,
        node_ids=node_ids,
        edge_masks=edge_mask,
        steps=steps,
        t0=t0,
        t1=t1
    )
    post_samples = post_samples[:, -theta_dim:]
    # Reshape to (B, num_samples, theta_dim)
    return post_samples.reshape(B, num_samples, theta_dim)