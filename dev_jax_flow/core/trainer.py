import os
import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax
from functools import partial
from train_interface.utils import marginalize, marginalize_node, save_params


class TrainFlowModel:
    def __init__(
        self,
        key,
        data,
        model_fn,
        params,
        nodes_max,
        condition_groups=None,
        batch_size=128,
        lr=1e-3,
        inner_train_loop_size=1000,
        early_stopping_patience=20,
        model_check_point_dir=None
    ):
        self.key = key
        self.data = data
        self.model_fn = model_fn
        self.params = params
        self.nodes_max = nodes_max
        self.node_ids = jnp.arange(nodes_max)
        self.condition_groups = condition_groups if condition_groups is not None else jnp.arange(nodes_max)
        self.unique_groups = jnp.unique(self.condition_groups)
        self.batch_size = batch_size
        self.inner_train_loop_size = inner_train_loop_size
        self.early_stopping_patience = early_stopping_patience
        self.model_check_point_dir = model_check_point_dir

        self.optimizer = optax.adam(lr)
        self.opt_state = self.optimizer.init(params)
        self.opt_params = None

    def random_batch(self, key):
        idx = jrandom.choice(key, self.data.shape[0], shape=(self.batch_size,))
        return self.data[idx]

    def build_groupwise_condition_mask(self, key, condition_prob=0.333):
        # Sample group-level Bernoulli draws for each item in batch
        rng_keys = jax.random.split(key, self.batch_size)

        def draw_group_mask(key):
            return jax.random.bernoulli(key, condition_prob, shape=(self.unique_groups.size,))  # (G,)

        group_draws = jax.vmap(draw_group_mask)(rng_keys)  # (B, G)
        # Map group draws to feature dimension
        mask = group_draws[:, self.condition_groups]  # (B, D)
        return mask.astype(bool)

    def build_groupwise_edge_mask(
            self,
            key: jax.random.PRNGKey,
            marginalize_prob: float = 0.5,
    ) -> jnp.ndarray:
        """
        Creates a group-wise edge mask (B, D, D) where entire rows/columns corresponding
        to marginalized groups are removed (set to False).
        """
        G = self.unique_groups.size
        # Decide for each sample and each group whether to KEEP that group
        keep_matrix = jax.random.bernoulli(key, 1.0 - marginalize_prob, shape=(self.batch_size//5, G))  # (B, G)
        # Create a binary mask for each group: (G, D)
        group_masks = jnp.stack([self.condition_groups == gid for gid in self.unique_groups], axis=0)  # (G, D)
        # Compute full dimension-wise masks: (B, D)
        keep_mask = jnp.dot(keep_matrix, group_masks.astype(jnp.int32)) > 0  # (B, D)
        # For each row: set edges to True if both dimensions are in retained groups
        edge_mask = jnp.einsum("bi,bj->bij", keep_mask, keep_mask)  # (B, D, D)
        # Remove masks for which no groups are kept
        edge_mask = jnp.where(jnp.any(edge_mask, axis=(1, 2), keepdims=True), edge_mask, False)
        # Fallback: if entire mask is False (no connection), restore full mask
        all_dropped = ~jnp.any(edge_mask, axis=(1, 2))  # (B,)
        edge_mask = jnp.where(all_dropped[:, None, None], jnp.ones_like(edge_mask), edge_mask)
        return edge_mask

    def loss_fn(self, params, key):
        batch_xs = self.random_batch(key)
        batch_xs_clean = batch_xs.copy()
        # fill nans with 0
        batch_xs_clean = jnp.nan_to_num(batch_xs_clean, 0.0)
        # Split the random key for different random operations
        rng_time, rng_sample, rng_data, rng_condition, rng_edge_mask1, rng_edge_mask2 = jax.random.split(key, 6)
        # Generate a batch of data using the generate_data function
        # batch_xs, batch_xs_clean = generate_data_combined_normal_par(key, batch_size, means, stds)
        # Create a binary condition mask indicating which nodes should be conditioned on
        # condition_mask = jax.random.bernoulli(rng_condition, 0.333, shape=(batch_xs.shape[0], batch_xs.shape[1]))
        condition_mask = self.build_groupwise_condition_mask(rng_condition, condition_prob=0.333)
        # Prevent conditioning on all nodes
        condition_mask_all_one = jnp.all(condition_mask, axis=-1, keepdims=True)
        condition_mask *= ~condition_mask_all_one
        # Expand dimensions of the condition mask for further computations
        condition_mask = condition_mask[..., None]
        # jax.debug.print("condition mask shape {}", condition_mask.shape)
        # jax.debug.print("condition mask is {}",condition_mask)
        # Create an initial dense edge mask (fully connected graph structure) for a subset of the batch
        edge_mask = jnp.ones((4 * self.batch_size // 5, batch_xs.shape[1], batch_xs.shape[1]), dtype=jnp.bool_)
        # Generate sparse masks by marginalizing over a subset of the batch
        marginal_mask = self.build_groupwise_edge_mask(
            key=rng_edge_mask1, marginalize_prob=0.2
        )
        # Concatenate the dense and sparse edge masks
        edge_masks = jnp.concatenate([edge_mask, marginal_mask], axis=0)
        # Randomly select between dense and sparse edge masks for each sample
        edge_masks = jax.random.choice(rng_edge_mask2, edge_masks, shape=(self.batch_size,), axis=0)
        # Apply marginalization based on NaN values in batch_xs to update the edge mask
        margarita = jax.vmap(marginalize)(batch_xs.squeeze(-1))  # Produces a mask with True where edges should be active
        # log the shape of the margarita mask
        edge_masks = edge_masks * margarita  # Apply the margarita mask to the edge mask)
        # jax.debug.print("edge mask shape {}", edge_masks.shape)
        # jax.debug.print("edge mask is {}",edge_masks)
        # Compute the loss using the denoising score matching function
        x0 = jrandom.normal(rng_sample, shape=(self.batch_size, self.nodes_max, 1))
        x1 = batch_xs_clean
        loss_mask = jnp.isnan(batch_xs)  # (B, T, 1), bool
        loss = flow_matching_loss(
            params,
            key,
            self.model_fn,
            x0,
            x1,
            self.node_ids,
            condition_mask,
            edge_masks,
            loss_mask,
            t_min=0.0,
            t_max=1.0
        )
        return loss

    @partial(jax.jit, static_argnums=(0,))
    def update(self, params, opt_state, key):
        loss, grads = jax.value_and_grad(self.loss_fn)(params, key)
        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return loss, new_params, new_opt_state

    def fit(self, epochs=100):
        best_loss = float('inf')
        best_params = None
        no_improve = 0
        key = self.key

        for epoch in range(epochs):
            epoch_loss = 0.0
            for _ in range(self.inner_train_loop_size):
                key, subkey = jrandom.split(key)
                loss, self.params, self.opt_state = self.update(self.params, self.opt_state, subkey)
                epoch_loss += loss / self.inner_train_loop_size

            print(f"Epoch {epoch+1}: loss = {epoch_loss:.6f}")

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_params = self.params
                no_improve = 0

                if isinstance(self.model_check_point_dir, str) and os.path.isdir(self.model_check_point_dir):
                    fname = os.path.join(self.model_check_point_dir, f"model_checkpoint_epoch_{epoch}.pkl")
                    save_params(best_params, fname)
                    print(f"Checkpoint saved to {fname}")
            else:
                no_improve += 1

            if no_improve >= self.early_stopping_patience:
                print("Stopping early.")
                break

        self.opt_params = best_params
        return best_params


def flow_matching_loss(params, key, model_fn, x0, x1, node_ids, condition_mask, edge_mask, loss_mask, t_min=0.0, t_max=1.0):
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
    cond_mask_bool = condition_mask.astype(bool)
    xt = jnp.where(cond_mask_bool, x1, xt)  # (B, T, 1)

    pred_velocity = model_fn(
        params,
        t,  # (B, 1)
        xt,  # (B, T, 1)
        node_ids,  # (B, T)
        condition_mask,  # (B, T)
        edge_mask=edge_mask
    )  # output: (B, T, 1)
    # Remove influence of condition mask and nan values on the loss
    loss_mask = cond_mask_bool | loss_mask
    # Compute the squared error loss
    velocity = jnp.nan_to_num(x1 - x0)
    pred_velocity = jnp.nan_to_num(pred_velocity)
    squared_error = (pred_velocity - velocity)**2
    squared_error = jnp.where(loss_mask, 0.0, squared_error)
    loss = jnp.sum(squared_error)
    norm = jnp.maximum(jnp.sum(~loss_mask), 1.0)
    return loss / norm



# Following lines shift the training to use a full batch instead of random batches.
    # def loss_fn(self, params, key, batch_xs):
    #     # batch_xs = self.random_batch(key)
    #     batch_xs_clean = batch_xs.copy()
    #     # fill nans with 0
    #     batch_xs_clean = jnp.nan_to_num(batch_xs_clean, 0.0)
    #     # Split the random key for different random operations
    #     rng_time, rng_sample, rng_data, rng_condition, rng_edge_mask1, rng_edge_mask2 = jax.random.split(key, 6)
    #     # Generate a batch of data using the generate_data function
    #     # batch_xs, batch_xs_clean = generate_data_combined_normal_par(key, batch_size, means, stds)
    #     # Create a binary condition mask indicating which nodes should be conditioned on
    #     condition_mask = jax.random.bernoulli(rng_condition, 0.333, shape=(batch_xs.shape[0], batch_xs.shape[1]))
    #     # Prevent conditioning on all nodes
    #     condition_mask_all_one = jnp.all(condition_mask, axis=-1, keepdims=True)
    #     condition_mask *= ~condition_mask_all_one
    #     # Expand dimensions of the condition mask for further computations
    #     condition_mask = condition_mask[..., None]
    #     # jax.debug.print("condition mask shape {}", condition_mask.shape)
    #     # jax.debug.print("condition mask is {}",condition_mask)
    #     # Create an initial dense edge mask (fully connected graph structure) for a subset of the batch
    #     edge_mask = jnp.ones((4 * self.batch_size // 5, batch_xs.shape[1], batch_xs.shape[1]), dtype=jnp.bool_)
    #     # Generate sparse masks by marginalizing over a subset of the batch
    #     marginal_mask = jax.vmap(marginalize_node, in_axes=(0, None))(
    #         jax.random.split(rng_edge_mask1, self.batch_size // 5), edge_mask[0])
    #     # Concatenate the dense and sparse edge masks
    #     edge_masks = jnp.concatenate([edge_mask, marginal_mask], axis=0)
    #     # Randomly select between dense and sparse edge masks for each sample
    #     edge_masks = jax.random.choice(rng_edge_mask2, edge_masks, shape=(self.batch_size,), axis=0)
    #     # Apply marginalization based on NaN values in batch_xs to update the edge mask
    #     margarita = jax.vmap(marginalize)(batch_xs.squeeze(-1))  # Produces a mask with True where edges should be active
    #     # log the shape of the margarita mask
    #     edge_masks = edge_masks * margarita  # Apply the margarita mask to the edge mask)
    #     # jax.debug.print("edge mask shape {}", edge_masks.shape)
    #     # jax.debug.print("edge mask is {}",edge_masks)
    #     # Compute the loss using the denoising score matching function
    #     x0 = jrandom.normal(rng_sample, shape=(self.batch_size, self.nodes_max, 1))
    #     x1 = batch_xs_clean
    #     loss_mask = jnp.isnan(batch_xs)  # (B, T, 1), bool
    #     loss = flow_matching_loss(
    #         params,
    #         key,
    #         self.model_fn,
    #         x0,
    #         x1,
    #         self.node_ids,
    #         condition_mask,
    #         edge_masks,
    #         loss_mask,
    #         t_min=0.0,
    #         t_max=1.0
    #     )
    #     return loss
    #
    # @partial(jax.jit, static_argnums=(0,))
    # def update(self, params, opt_state, key, batch):
    #     loss, grads = jax.value_and_grad(self.loss_fn)(params, key, batch)
    #     updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
    #     new_params = optax.apply_updates(params, updates)
    #     return loss, new_params, new_opt_state
    #
    # def fit(self, epochs=100):
    #     best_loss = float('inf')
    #     best_params = None
    #     no_improve = 0
    #     key = self.key
    #     N = self.data.shape[0]
    #
    #     for epoch in range(epochs):
    #         key, shuffle_key = jrandom.split(key)
    #         perm = jrandom.permutation(shuffle_key, N)
    #         data_shuffled = self.data[perm]
    #
    #         epoch_loss = 0.0
    #         num_batches = (N + self.batch_size - 1) // self.batch_size
    #
    #         for i in range(0, N, self.batch_size):
    #             key, subkey = jrandom.split(key)
    #             batch = data_shuffled[i:i + self.batch_size]
    #             # loss, self.params, self.opt_state = self.update(self.params, self.opt_state, subkey)
    #             loss, self.params, self.opt_state = self.update(self.params, self.opt_state, subkey, batch)
    #             epoch_loss += loss / num_batches
    #
    #         print(f"Epoch {epoch+1}: loss = {epoch_loss:.6f}")
    #
    #         if epoch_loss < best_loss:
    #             best_loss = epoch_loss
    #             best_params = self.params
    #             no_improve = 0
    #
    #             if isinstance(self.model_check_point_dir, str) and os.path.isdir(self.model_check_point_dir):
    #                 fname = os.path.join(self.model_check_point_dir, f"model_checkpoint_epoch_{epoch}.pkl")
    #                 save_params(best_params, fname)
    #                 print(f"Checkpoint saved to {fname}")
    #         else:
    #             no_improve += 1
    #
    #         if no_improve >= self.early_stopping_patience:
    #             print("Stopping early.")
    #             break
    #
    #     self.opt_params = best_params
    #     return best_params
