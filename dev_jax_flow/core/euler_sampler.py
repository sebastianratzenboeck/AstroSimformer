from functools import partial
from typing import Callable, Optional, Sequence
import jax
import jax.numpy as jnp


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
        x = x_flat.reshape(-1, len(self.node_ids))
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
    shape,                # (B,)
    condition_mask: jnp.ndarray,        # (B, M) or (B, M, D_mask)
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
    condition_mask = condition_mask[:len(node_ids)]
    condition_values = condition_values[:len(node_ids)]
    x0 = jax.random.normal(key, shape + (len(node_ids),))

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
    static_argnames=("model_fn", "shape", "steps"),
)