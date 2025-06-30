import os
import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax
import haiku as hk
from haiku.nets import MLP
from functools import partial
from train_interface.utils import marginalize, marginalize_node, save_params


def make_autoencoder(in_dim, hidden_dim):
    def forward(x):
        encoder = MLP([hidden_dim], activation=jax.nn.relu, activate_final=True, name="encoder")
        decoder = MLP([hidden_dim, in_dim], activation=jax.nn.relu, activate_final=False, name="decoder")
        z = encoder(x)
        x_hat = decoder(z)
        return x_hat
    return hk.without_apply_rng(hk.transform(forward))


<<<<<<< Updated upstream
def encode_batch_fn(batch, encode_mask, encoder_apply, encoder_params):
    x_encoded = batch[:, encode_mask, :]
=======
def blend_stop_gradient(z, alpha):
    return jax.lax.stop_gradient(z) * (1 - alpha) + z * alpha

def smooth_freeze_schedule(epoch, freeze_epoch=10, sharpness=2.0):
    alpha = 1 / (1 + jnp.exp((epoch - freeze_epoch) * sharpness)) 
    print(f'Current alpha={alpha}')
    return alpha


def encode_batch_fn(batch, encode_mask, encoder_apply, encoder_params, alpha):
    x_encoded = batch[:, encode_mask]
>>>>>>> Stashed changes
    if x_encoded.ndim == 3 and x_encoded.shape[-1] == 1:
        x_encoded = x_encoded[..., 0]
    x_pass = batch[:, ~encode_mask, :]
    if x_pass.ndim == 3 and x_pass.shape[-1] == 1:
        x_pass = x_pass[..., 0]
    x_hat = encoder_apply(encoder_params, x_encoded)
    # jax.debug.print("mse {}", jnp.mean(jnp.square(x_pass - x_hat)))
    full_input = jnp.concatenate([x_hat, x_pass], axis=-1)
    return full_input[..., None], jnp.mean(jnp.square(x_pass - x_hat))


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
        alpha=0.2,
        freeze_encoder_epoch=10
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
        self.freeze_encoder_epoch = freeze_encoder_epoch
        self.current_epoch = 0

        self.optimizer = optax.adam(lr)
        self.opt_state = self.optimizer.init(params)
        self.opt_params = None

        in_dim = int(jnp.sum(self.encode_mask))
        self.encoder_fn = make_autoencoder(in_dim, encoder_hidden_dim)
        self.encoder_params = self.encoder_fn.init(key, jnp.zeros((1, in_dim)))
        self.encoder_optimizer = optax.adam(lr)
        self.encoder_opt_state = self.encoder_optimizer.init(self.encoder_params)

    def freeze_encoder_decoder(self):
        return self.current_epoch >= self.freeze_encoder_epoch

    def linear_alpha_schedule(self, alpha_start=0.0, alpha_end=1.0, warmup_epochs=10):
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

    def loss_fn(self, params, encoder_params, key, alpha):
        batch_raw = self.random_batch(key)
        batch_xs, mse_loss = encode_batch_fn(
            batch_raw,
            self.encode_mask,
            self.encoder_fn.apply,
<<<<<<< Updated upstream
            encoder_params
=======
            self.encoder_params,
            smooth_freeze_schedule(self.current_epoch, 40, 0.25)
>>>>>>> Stashed changes
        )
        batch_xs_clean = batch_xs.copy()
        rng_time, rng_sample, rng_data, rng_condition, rng_edge_mask1, rng_edge_mask2 = jax.random.split(key, 6)

        condition_mask = self.build_groupwise_condition_mask(rng_condition, condition_prob=0.333)
        condition_mask_all_one = jnp.all(condition_mask, axis=-1, keepdims=True)
        condition_mask *= ~condition_mask_all_one
        condition_mask = condition_mask[..., None]

        edge_masks = jnp.ones((self.batch_size, batch_xs.shape[1], batch_xs.shape[1]), dtype=jnp.bool_)

        x0 = jrandom.normal(rng_sample, shape=(self.batch_size, self.nodes_max, 1))
        x1 = jnp.nan_to_num(batch_xs_clean, 0.0)
        loss_mask = jnp.isnan(batch_xs)

        flow_loss = flow_matching_loss(
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
        total_loss = mse_loss + flow_loss
        return total_loss, (mse_loss, flow_loss)

    @partial(jax.jit, static_argnums=(0,))
    def update(self, params, encoder_params, opt_state, encoder_opt_state, key, alpha):
        (loss, (mse_loss, flow_loss)), grads = jax.value_and_grad(
            self.loss_fn, has_aux=True, argnums=(0, 1))(params, encoder_params, key, alpha)
        param_grads, encoder_grads = grads

        updates, new_opt_state = self.optimizer.update(param_grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        if not self.freeze_encoder_decoder():
            encoder_updates, new_encoder_opt_state = self.encoder_optimizer.update(encoder_grads, encoder_opt_state, encoder_params)
            new_encoder_params = optax.apply_updates(encoder_params, encoder_updates)
        else:
            new_encoder_opt_state = encoder_opt_state
            new_encoder_params = encoder_params

        return loss, new_params, new_opt_state, new_encoder_params, new_encoder_opt_state

    def fit(self, epochs=100):
        best_loss = float('inf')
        best_params = None
        no_improve = 0
        key = self.key

        for epoch in range(epochs):
            self.current_epoch = epoch
            alpha = self.linear_alpha_schedule()
            epoch_loss = 0.0
            for _ in range(self.inner_train_loop_size):
                key, subkey = jrandom.split(key)
                loss, self.params, self.opt_state, self.encoder_params, self.encoder_opt_state = self.update(
                    self.params, self.encoder_params, self.opt_state, self.encoder_opt_state, subkey, alpha)
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
