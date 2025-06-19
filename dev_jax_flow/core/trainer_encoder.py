import os
import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax
import haiku as hk
from haiku.nets import MLP
from functools import partial
from train_interface.utils import marginalize, marginalize_node, save_params


def make_encoder(in_dim, hidden_dim, out_dim):
    def forward(x):
        return MLP([hidden_dim, out_dim], activation=jax.nn.sigmoid, activate_final=True)(x)
    return hk.without_apply_rng(hk.transform(forward))


def blend_stop_gradient(z, alpha):
    return jax.lax.stop_gradient(z) * (1 - alpha) + z * alpha


def encode_batch_fn(batch, encode_mask, encoder_apply, encoder_params, alpha):
    x_encoded = batch[:, encode_mask]
    if x_encoded.ndim == 3 and x_encoded.shape[-1] == 1:
        x_encoded = x_encoded[..., 0]  # Fix trailing singleton dim
    x_pass = batch[:, ~encode_mask]
    if x_pass.ndim == 3 and x_pass.shape[-1] == 1:
        x_pass = x_pass[..., 0]
    z = encoder_apply(encoder_params, x_encoded)
    z_stable = blend_stop_gradient(z, alpha)
    return jnp.concatenate([z_stable, x_pass], axis=-1)[..., None]  # Add trailing singleton dim


class TrainFlowModel:
    def __init__(
        self,
        key,
        data,
        model_fn,
        params,
        nodes_max,
        condition_groups=None,
        encode_mask=None,
        batch_size=128,
        lr=1e-3,
        inner_train_loop_size=1000,
        early_stopping_patience=20,
        model_check_point_dir=None,
        encoder_hidden_dim=64,
        alpha=0.2
    ):
        self.key = key
        self.data = data
        self.model_fn = model_fn
        self.params = params
        self.nodes_max = nodes_max
        self.node_ids = jnp.arange(nodes_max)
        self.condition_groups = condition_groups if condition_groups is not None else jnp.arange(nodes_max)
        self.unique_groups = jnp.unique(self.condition_groups)
        self.encode_mask = encode_mask if encode_mask is not None else jnp.zeros((nodes_max,), dtype=bool)
        self.batch_size = batch_size
        self.inner_train_loop_size = inner_train_loop_size
        self.early_stopping_patience = early_stopping_patience
        self.model_check_point_dir = model_check_point_dir
        self.alpha = alpha
        self.current_epoch = 0

        self.optimizer = optax.adam(lr)
        self.opt_state = self.optimizer.init(params)
        self.opt_params = None

        # Setup encoder and EMA buffer
        in_dim = int(jnp.sum(self.encode_mask))
        print("in_dim ", in_dim)
        self.encoder_fn = make_encoder(in_dim, encoder_hidden_dim, in_dim)
        self.encoder_params = self.encoder_fn.init(key, jnp.zeros((1, in_dim)))

    def linear_alpha_schedule(self, alpha_start=0.0, alpha_end=1.0, warmup_epochs=10):
        """Linearly increase alpha from alpha_start to alpha_end over warmup_epochs."""
        if self.current_epoch >= warmup_epochs:
            return alpha_end
        return alpha_start + (alpha_end - alpha_start) * (self.current_epoch / warmup_epochs)

    def random_batch(self, key):
        idx = jrandom.choice(key, self.data.shape[0], shape=(self.batch_size,), replace=False)
        return self.data[idx]

    def build_groupwise_condition_mask(self, key, condition_prob=0.333):
        rng_keys = jax.random.split(key, self.batch_size)

        def draw_group_mask(key):
            return jax.random.bernoulli(key, condition_prob, shape=(self.unique_groups.size,))

        group_draws = jax.vmap(draw_group_mask)(rng_keys)
        mask = group_draws[:, self.condition_groups]
        return mask.astype(bool)

    def build_groupwise_edge_mask(self, key: jax.random.PRNGKey, marginalize_prob: float = 0.5) -> jnp.ndarray:
        G = self.unique_groups.size
        keep_matrix = jax.random.bernoulli(key, 1.0 - marginalize_prob, shape=(self.batch_size//5, G))
        group_masks = jnp.stack([self.condition_groups == gid for gid in self.unique_groups], axis=0)
        keep_mask = jnp.dot(keep_matrix, group_masks.astype(jnp.int32)) > 0
        edge_mask = jnp.einsum("bi,bj->bij", keep_mask, keep_mask)
        edge_mask = jnp.where(jnp.any(edge_mask, axis=(1, 2), keepdims=True), edge_mask, False)
        all_dropped = ~jnp.any(edge_mask, axis=(1, 2))
        edge_mask = jnp.where(all_dropped[:, None, None], jnp.ones_like(edge_mask), edge_mask)
        return edge_mask

    def loss_fn(self, params, key):
        batch_xs = self.random_batch(key)
        print("batch_xs.shape ", batch_xs.shape)
        batch_xs = encode_batch_fn(
            batch_xs,
            self.encode_mask,
            self.encoder_fn.apply,
            self.encoder_params,
            self.linear_alpha_schedule()
        )
        batch_xs_clean = batch_xs.copy()
        rng_time, rng_sample, rng_data, rng_condition, rng_edge_mask1, rng_edge_mask2 = jax.random.split(key, 6)
        # condition_mask = self.build_groupwise_condition_mask(rng_condition, condition_prob=0.333)
        # condition_mask_all_one = jnp.all(condition_mask, axis=-1, keepdims=True)
        # condition_mask *= ~condition_mask_all_one
        # ----- new condition mask -----
        # Condition on the encoding mask
        condition_mask = jnp.broadcast_to(self.encode_mask, (self.batch_size, self.nodes_max))[..., None].astype(int)
        # ------------------------------
        # ---- Edge mask for marginalization ----
        # edge_mask = jnp.ones((4*self.batch_size//5, batch_xs.shape[1], batch_xs.shape[1]), dtype=jnp.bool_)
        # marginal_mask = self.build_groupwise_edge_mask(key=rng_edge_mask1, marginalize_prob=0.1)
        # edge_masks = jnp.concatenate([edge_mask, marginal_mask], axis=0)
        # edge_masks = jax.random.choice(rng_edge_mask2, edge_masks, shape=(self.batch_size,), axis=0)
        # margarita = jax.vmap(marginalize)(batch_xs.squeeze(-1))
        # edge_masks = edge_masks * margarita
        # ----- New edge mask -----
        edge_masks = jnp.ones((self.batch_size, self.nodes_max, self.nodes_max), dtype=jnp.bool_)
        # ----------------------------------------
        x0 = jrandom.normal(rng_sample, shape=(self.batch_size, self.nodes_max, 1))
        x1 = jnp.nan_to_num(batch_xs_clean, 0.0)
        loss_mask = jnp.isnan(batch_xs)
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
            self.current_epoch = epoch
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
    B, T, _ = x0.shape
    assert x0.shape == x1.shape, "x0 and x1 must have the same shape"

    key, subkey = jax.random.split(key)
    t = jax.random.uniform(subkey, shape=(B, 1), minval=t_min, maxval=t_max)
    t = t[:, None, :]
    xt = (1 - t) * x0 + t * x1
    cond_mask_bool = condition_mask.astype(bool)
    xt = jnp.where(cond_mask_bool, x1, xt)

    pred_velocity = model_fn(
        params,
        t,
        xt,
        node_ids,
        condition_mask,
        edge_mask=edge_mask
    )
    loss_mask = cond_mask_bool | loss_mask
    velocity = jnp.nan_to_num(x1 - x0)
    pred_velocity = jnp.nan_to_num(pred_velocity)
    squared_error = (pred_velocity - velocity)**2
    squared_error = jnp.where(loss_mask, 0.0, squared_error)
    loss = jnp.sum(squared_error)
    norm = jnp.maximum(jnp.sum(~loss_mask), 1.0)
    return loss / norm
