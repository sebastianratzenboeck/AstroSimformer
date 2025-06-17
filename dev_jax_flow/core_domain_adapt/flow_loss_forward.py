import jax
import jax.numpy as jnp
import jax.random as jrandom
from train_interface.utils import marginalize, marginalize_node



class SimformerLossWrapper:
    def __init__(
        self,
        model_fn,
    ):
        self.model_fn = model_fn

    def forward(self, params, x, key):
        if x.ndim == 2:
            x = x[..., None]
        B, T, _ = x.shape
        node_ids = jnp.arange(T)
        # Create a mask for NaN values in the input data
        loss_mask = jnp.isnan(x)
        x1 = x.copy()
        # fill nans with 0
        x1 = jnp.nan_to_num(x1, 0.0)
        # Split the random key for different random operations
        rng_time, rng_sample, rng_data, rng_condition, rng_edge_mask1, rng_edge_mask2 = jax.random.split(key, 6)
        # Generate a batch of data using the generate_data function
        # batch_xs, batch_xs_clean = generate_data_combined_normal_par(key, batch_size, means, stds)
        # Create a binary condition mask indicating which nodes should be conditioned on
        condition_mask = jax.random.bernoulli(rng_condition, 0.333, shape=(x.shape[0], x.shape[1]))
        # Prevent conditioning on all nodes
        condition_mask_all_one = jnp.all(condition_mask, axis=-1, keepdims=True)
        condition_mask *= ~condition_mask_all_one
        # Expand dimensions of the condition mask for further computations
        condition_mask = condition_mask[..., None]
        # jax.debug.print("condition mask shape {}", condition_mask.shape)
        # jax.debug.print("condition mask is {}",condition_mask)
        # Create an initial dense edge mask (fully connected graph structure) for a subset of the batch
        edge_mask = jnp.ones((4 * B // 5, x.shape[1], x.shape[1]), dtype=jnp.bool_)
        # Generate sparse masks by marginalizing over a subset of the batch
        marginal_mask = jax.vmap(marginalize_node, in_axes=(0, None))(
            jax.random.split(rng_edge_mask1, B // 5), edge_mask[0]
        )
        # Concatenate the dense and sparse edge masks
        edge_masks = jnp.concatenate([edge_mask, marginal_mask], axis=0)
        # Randomly select between dense and sparse edge masks for each sample
        # edge_masks = jax.random.choice(rng_edge_mask2, edge_masks, shape=(B,), axis=0)
        # Above line not always XLA-traceable, the following is JIT-safe & differentiable, and functionally equivalent
        indices = jax.random.randint(rng_edge_mask2, shape=(B,), minval=0, maxval=edge_masks.shape[0])
        edge_masks = edge_masks[indices]
        # Apply marginalization based on NaN values in batch_xs to update the edge mask
        margarita = jax.vmap(marginalize)(x.squeeze(-1))  # Produces a mask with True where edges should be active
        # log the shape of the margarita mask
        edge_masks = edge_masks * margarita  # Apply the margarita mask to the edge mask)
        # jax.debug.print("edge mask shape {}", edge_masks.shape)
        # jax.debug.print("edge mask is {}",edge_masks)
        # Compute the loss using the denoising score matching function
        x0 = jrandom.normal(rng_sample, shape=x1.shape)
        loss = flow_matching_loss(
            params,
            key,
            self.model_fn,
            x0,
            x1,
            node_ids,
            condition_mask,
            edge_masks,
            loss_mask,
            t_min=0.0,
            t_max=1.0
        )
        return loss


def flow_matching_loss(
        params, key, model_fn, x0, x1, node_ids, condition_mask, edge_mask, loss_mask, t_min=0.0, t_max=1.0
):
    """
    Computes flow matching loss for Simformer in JAX.

    Args:
        params: model parameters (Haiku)
        key: jax PRNG key
        model_fn: model(params, t, x, node_ids, condition_mask, edge_mask)
        x0, x1: (B, T) pairs of source/target samples
        node_ids: (B, T)
        condition_mask: (B, T)
        edge_mask: (B, T, T)
        t_min, t_max: time sampling range

    Returns:
        scalar loss
    """
    B, T, _ = x0.shape
    assert x0.shape == x1.shape, "x0 and x1 must have the same shape"

    key, subkey = jax.random.split(key)
    t = jax.random.uniform(subkey, shape=(B, 1), minval=t_min, maxval=t_max)
    t = t[:, None, :]  # shape (B, 1, 1)
    xt = (1 - t) * x0 + t * x1
    # xt = xt * (1 - condition_mask) + x1 * condition_mask
    # cond_mask_bool = condition_mask.astype(bool)  # <- should already be a bool array
    xt = jnp.where(condition_mask, x1, xt)  # (B, T, 1)

    pred_velocity = model_fn(
        params,
        t,  # (B, 1)
        xt,  # (B, T, 1)
        node_ids,  # (B, T)
        condition_mask,  # (B, T)
        edge_mask=edge_mask
    )  # output: (B, T, 1)
    # Remove influence of condition mask and nan values on the loss
    loss_mask = condition_mask | loss_mask
    # Compute the squared error loss
    velocity = jnp.nan_to_num(x1 - x0)
    pred_velocity = jnp.nan_to_num(pred_velocity)
    squared_error = (pred_velocity - velocity)**2
    squared_error = jnp.where(loss_mask, 0.0, squared_error)
    loss = jnp.sum(squared_error)
    norm = jnp.maximum(jnp.sum(~loss_mask), 1.0)
    return loss / norm