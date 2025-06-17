import os
import jax
import jax.numpy as jnp
import jax.random as jrandom
from typing import Optional, Callable
from jaxtyping import PyTree, Array
import optax
from functools import partial
from utils import marginalize, marginalize_node
from utils import save_params, load_params, file_with_largest_integer



class TrainScoreModel:
    def __init__(
            self, key: jrandom.PRNGKey,
            data, sde, model_fn, params, T_min, nodes_max,
            batch_size=1024, lr=1e-3, inner_train_loop_size=1_000,
            early_stopping_patience=20,
            model_check_point_dir=None
    ):
        self.key = key
        self.data = data
        # SDE
        self.sde = sde
        self.model_fn = model_fn
        # Training parameters
        self.T_min = T_min
        self.nodes_max = nodes_max
        self.node_ids = jnp.arange(nodes_max)
        self.batch_size = batch_size
        self.inner_train_loop_size = inner_train_loop_size
        self.early_stopping_patience = early_stopping_patience
        self.model_check_point_dir = model_check_point_dir
        # This part is training the transformer model until a good result is reached which does not improve for some iterations
        # Assuming optax optimizer, params, and loss_fn are already defined
        self.params = params
        self.optimizer = optax.adam(lr)
        self.opt_state = self.optimizer.init(params)
        self.opt_params = None

    def get_sde(self):
        return self.sde

    def get_model_fn(self):
        return self.model_fn

    def get_opt_params(self):
        return self.opt_params

    def load_best_model(self, file_path=None):
        if file_path is not None:
            # Load the best parameters
            self.opt_params = load_params(file_path)
            print(f"Best parameters loaded from {file_path}")
        elif isinstance(self.model_check_point_dir, str):
            # Load the best parameters
            fname = file_with_largest_integer(
                folder_path=self.model_check_point_dir, file_pattern="model_checkpoint_epoch_*.pkl"
            )
            if fname is not None:
                self.opt_params = load_params(fname)
                print(f"Best parameters loaded from {fname}")
            else:
                raise ValueError("No model checkpoint found in the specified directory.")
        else:
            raise ValueError("Model checkpoint directory is not set.")

    @staticmethod
    def denoising_score_matching_loss(
        params: PyTree,
        key: jrandom.PRNGKey,
        times: Array,
        xs_target: Array,
        xs_target_clean: Array,
        loss_mask: Optional[Array],
        *args,
        model_fn: Callable,
        mean_fn: Callable,
        std_fn: Callable,
        weight_fn: Callable,
        axis: int = -2,
        rebalance_loss: bool = False,
        **kwargs,
    ) -> Array:
        """This function computes the denoising score matching loss. Which can be used to train diffusion models.
        modified matching loss function in order to work with nan values
        Args:
            params (PyTree): Parameters of the model_fn given as a PyTree.
            key (PRNGKey): Random generator key.
            times (Array): Time points, should be broadcastable to shape (batch_size, 1).
            xs_target (Array): Target distribution.
            loss_mask (Optional[Array]): Mask for the target distribution. If None, no mask is applied, should be broadcastable to shape (batch_size, 1).
            model_fn (Callable): Score model that takes parameters, times, and samples as input and returns the score. Should be a function of the form model_fn(params, times, xs_t, *args) -> s_t.
            mean_fn (Callable): Mean function of the SDE.
            std_fn (Callable): Std function of the SDE.
            weight_fn (Callable): Weight function for the loss.
            axis (int, optional): Axis to sum over. Defaults to -2.
            rebalance_loss (bool, optional): Whether to rebalance the loss by the number of valid elements. Defaults to False.

        Returns:
            Array: Loss
        """
        eps = jax.random.normal(key, shape=xs_target.shape)
        mean_t = mean_fn(times, xs_target_clean)
        std_t = std_fn(times, xs_target_clean)
        xs_t = mean_t + std_t * eps

        # TODO: This was removed by Alex, maybe it should be added back
        if loss_mask is not None:
            loss_maks = loss_mask.astype(bool)
            xs_t = jnp.where(loss_maks, xs_target_clean, xs_t)

        # jax.debug.print("mean_t is {}",mean_t.shape)
        # jax.debug.print("std_t is {}",std_t.shape)
        score_pred = model_fn(params, times, xs_t, *args, **kwargs)
        score_target = -eps / std_t
        # jax.debug.print("score_pred is {}",score_pred)
        # jax.debug.print("score_target is {}",score_target)
        # Compute the squared error
        loss = (score_pred - score_target) ** 2
        # Apply the combined mask to the loss: add NaN mask
        loss_mask = loss_mask | jnp.isnan(xs_target)
        loss = jnp.where(loss_mask, 0.0, loss)
        # Apply weighting and sum over specified axis
        loss = weight_fn(times) * jnp.sum(loss, axis=axis, keepdims=True)
        # Optional rebalancing
        # if rebalance_loss:
        num_elements = jnp.sum(~loss_mask, axis=axis, keepdims=True)
        loss = jnp.where(num_elements > 0, loss / num_elements, 0.0)
        # Mean over batch
        loss = jnp.mean(loss)
        return loss

    def weight_fn(self, t: Array):
        # Calculate weighting based on the diffusion process at time 't'.
        # sde.diffusion(t, jnp.ones((1,1,1))) computes diffusion for all-ones input.
        # Squared and clipped to ensure it's bounded by 1e-4.
        return jnp.clip(self.sde.diffusion(t, jnp.ones((1,1,1)))**2, 1e-4)

    def data_loader(self, key: jrandom.PRNGKey):
        """
        Yields batches of data. If shuffle is True, data will be randomly permuted.

        Parameters:
            data (jnp.ndarray): Dataset to load batches from.
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle the data before batching.
            key (jax.random.PRNGKey): Random key for shuffling.
        """
        # Shuffle the data
        permuted_indices = jrandom.permutation(key, len(self.data))
        train_data = self.data[permuted_indices]
        # yield batches
        n_samples = len(train_data)
        for start in range(0, n_samples, self.batch_size):
            yield train_data[start: start + self.batch_size]

    def random_batch(self, key: jrandom.PRNGKey):
        """
        Yields a random batch of data from the training set.

        Parameters:
            key (jax.random.PRNGKey): Random key for generating random indices.
        """
        # Generate random indices
        indices = jrandom.choice(key, jnp.arange(len(self.data)), shape=(self.batch_size,))
        # Yield the corresponding batch
        return self.data[indices]

    # def loss_fn(self, params: dict, key: jrandom.PRNGKey, batch_xs: Array):
    def loss_fn(self, params: dict, key: jrandom.PRNGKey):
        batch_xs = self.random_batch(key)
        batch_xs_clean = batch_xs.copy()
        # fill nans with 0
        batch_xs_clean = jnp.nan_to_num(batch_xs_clean, 0.0)
        # Split the random key for different random operations
        rng_time, rng_sample, rng_data, rng_condition, rng_edge_mask1, rng_edge_mask2 = jax.random.split(key, 6)
        # Generate random times between T_min and 1.0 for each sample in the batch
        times = jax.random.uniform(rng_time, (self.batch_size, 1, 1), minval=self.T_min, maxval=1.0)
        # Generate a batch of data using the generate_data function
        # batch_xs, batch_xs_clean = generate_data_combined_normal_par(key, batch_size, means, stds)
        # Create a binary condition mask indicating which nodes should be conditioned on
        condition_mask = jax.random.bernoulli(rng_condition, 0.333, shape=(batch_xs.shape[0], batch_xs.shape[1]))
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
        marginal_mask = jax.vmap(marginalize_node, in_axes=(0, None))(
            jax.random.split(rng_edge_mask1, self.batch_size // 5), edge_mask[0])
        # Concatenate the dense and sparse edge masks
        edge_masks = jnp.concatenate([edge_mask, marginal_mask], axis=0)
        # Randomly select between dense and sparse edge masks for each sample
        edge_masks = jax.random.choice(rng_edge_mask2, edge_masks, shape=(self.batch_size,), axis=0)
        # Apply marginalization based on NaN values in batch_xs to update the edge mask
        margarita = jax.vmap(marginalize)(batch_xs).squeeze(-1)  # Produces a mask with True where edges should be active
        edge_masks = edge_masks * margarita  # Apply the margarita mask to the edge mask)
        # jax.debug.print("edge mask shape {}", edge_masks.shape)
        # jax.debug.print("edge mask is {}",edge_masks)
        # Compute the loss using the denoising score matching function
        loss = self.denoising_score_matching_loss(
            params, rng_sample, times, batch_xs, batch_xs_clean, condition_mask,
            model_fn=self.model_fn,
            mean_fn=self.sde.marginal_mean,
            std_fn=self.sde.marginal_stddev,
            weight_fn=self.weight_fn,
            # Node IDs to identify the nodes in the graph (assumes `node_ids` is predefined)
            node_ids=self.node_ids,
            condition_mask=condition_mask,
            edge_mask=edge_masks
        )
        return loss

    @partial(jax.pmap, axis_name="num_devices", static_broadcasted_argnums=(0,))
    def update(self, params, rng, opt_state):
    # def update(self, params, batch, rng, opt_state):
    #     loss, grads = jax.value_and_grad(self.loss_fn)(params, rng, batch)
        loss, grads = jax.value_and_grad(self.loss_fn)(params, rng)
        # Average loss and gradients across devices
        loss = jax.lax.pmean(loss, axis_name="num_devices")
        grads = jax.lax.pmean(grads, axis_name="num_devices")
        # jax.debug.print("loss is {}",loss)
        # jax.debug.print("grads is {}",grads)
        # Apply optimizer updates
        updates, opt_state = self.optimizer.update(grads, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        return loss, params, opt_state

    def fit(self, epochs: int=1_000):
        n_devices = jax.local_device_count()
        # jax.debug.print("Number of devices: {}", n_devices)
        replicated_params = jax.tree_util.tree_map(lambda x: jnp.array([x] * n_devices), self.params)
        replicated_opt_state = jax.tree_util.tree_map(lambda x: jnp.array([x] * n_devices), self.opt_state)
        # Initialize variables for early stopping
        best_loss = float('inf')
        no_improvement_counter = 0
        best_params = None
        # Initialize the random key
        key = self.key
        # create train loader
        for epoch in range(epochs):
            epoch_loss = 0
            # create new key
            key, subkey = jrandom.split(key)
            # train_loader = self.data_loader(key=subkey)
            # for batch in train_loader:
            for _ in range(self.inner_train_loop_size):
                # for pmap to work with batch size, the batch size should be divisible by the number of devices
                # if batch.shape[0] == self.batch_size:
                key, subkey = jrandom.split(key)
                # batch = batch.reshape((n_devices, -1, *batch.shape[1:]))
                loss, replicated_params, replicated_opt_state = self.update(
                    replicated_params,
                    # batch,
                    jax.random.split(subkey, n_devices),
                    replicated_opt_state
                )
                # jax.debug.print("loss {}",loss)
                # Average loss for this step and accumulate for epoch
                epoch_loss += loss[0]/self.inner_train_loop_size
                # n_loops += 1
            # epoch_loss /= n_loops
            # Print the epoch loss
            print(f"Epoch {epoch + 1} loss: {epoch_loss}")
            # Check for improvement
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                no_improvement_counter = 0
                best_params = jax.tree_util.tree_map(lambda x: x.copy(), replicated_params)
                # ---- Model checkpointing ----
                if isinstance(self.model_check_point_dir, str):
                    if os.path.isdir(self.model_check_point_dir) and (epoch >= 5):
                        fname = os.path.join(self.model_check_point_dir, f"model_checkpoint_epoch_{epoch}.pkl")
                        # Save the best parameters
                        best_params_current = jax.jax.tree_util.tree_map(lambda x: x[0], best_params)
                        save_params(best_params_current, fname)
                        print(f"Best parameters saved at epoch {epoch + 1}")
                # ---- End of model checkpointing ----
            else:
                no_improvement_counter += 1
            # Stop if no improvement over 5 iterations
            if no_improvement_counter >= self.early_stopping_patience:
                print("Stopping early due to no improvement.")
                break
        # Retrieve final trained parameters (take the best parameters)
        self.opt_params = jax.tree_util.tree_map(lambda x: x[0], best_params)
        return