import os
import jax
import jax.random as jrandom
import optax
from flax.training.train_state import TrainState
from flax.serialization import to_bytes
from typing import Dict, Any, Callable, Tuple
from SimflowerMultimodal import SimformerMultimodalFlowModel


class TrainMultimodalSimformer:
    def __init__(
        self,
        key,
        domain_adapter,
        model_fn,
        params,
        flow_matching_loss_fn,
        config,
        get_batch_fn,
        inner_train_loop_size=1000,
        early_stopping_patience=20,
        model_check_point_dir=None
    ):
        self.key = key
        self.model_check_point_dir = model_check_point_dir
        self.get_batch_fn = get_batch_fn
        self.inner_train_loop_size = inner_train_loop_size
        self.early_stopping_patience = early_stopping_patience
        self.best_params = None

        self.flow_model = SimformerMultimodalFlowModel(
            domain_adapter=domain_adapter,
            loss_wrapper=flow_matching_loss_fn,
            config=config
        )

        self.train_state = TrainState.create(
            apply_fn=model_fn,
            params=params,
            tx=optax.adam(config.get("lr", 1e-3))
        )

    def fit(self, epochs=100):
        best_loss = float("inf")
        self.best_params = None
        no_improve = 0
        key = self.key

        for epoch in range(epochs):
            epoch_loss = 0.0
            logs_accum = {}
            for _ in range(self.inner_train_loop_size):
                key, subkey = jrandom.split(key)
                batch, key = self.get_batch_fn(key)
                self.train_state, logs = self.flow_model.train_step(self.train_state, batch, subkey)
                loss = logs["flow_matching"] + sum(logs.get(k, 0.0) for k in logs if k != "flow_matching")
                epoch_loss += loss / self.inner_train_loop_size

                for k, v in logs.items():
                    logs_accum[k] = logs_accum.get(k, 0.0) + v / self.inner_train_loop_size

            print(f"Epoch {epoch+1}: loss = {epoch_loss:.6f}, details = {logs_accum}")

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                self.best_params = self.train_state.params
                no_improve = 0

                if isinstance(self.model_check_point_dir, str) and os.path.isdir(self.model_check_point_dir):
                    fname = os.path.join(self.model_check_point_dir, f"model_checkpoint_epoch_{epoch}.pkl")
                    with open(fname, "wb") as f:
                        f.write(to_bytes(self.best_params))
                    print(f"Checkpoint saved to {fname}")
            else:
                no_improve += 1

            if no_improve >= self.early_stopping_patience:
                print("Stopping early.")
                break

        return self.best_params


def get_batch_fn_factory(
    dataset: Dict[str, Any],
    modalities: list,
    sim_theta_key: str = "theta",
    batch_size: int = 128
) -> Callable[[jax.random.PRNGKey], Tuple[Dict[str, Any], jax.random.PRNGKey]]:
    """
    Returns a stateless, JAX-compatible batch sampling function.

    Args:
        dataset: dict with keys 'sim', 'real', 'pair'
        modalities: list of modality names
        sim_theta_key: name of theta key in dataset['sim']
        batch_size: number of samples

    Returns:
        A function that takes a PRNGKey and returns a batch and next PRNGKey
    """
    N_sim = dataset['sim'][modalities[0]].shape[0]
    N_real = dataset['real'][modalities[0]].shape[0]
    N_pair = dataset['pair'][modalities[0]][0].shape[0]

    def get_batch(rng: jax.random.PRNGKey) -> Tuple[Dict[str, Any], jax.random.PRNGKey]:
        rng, sk_sim, sk_real, sk_pair = jax.random.split(rng, 4)

        idx_sim = jax.random.choice(sk_sim, N_sim, shape=(batch_size,), replace=False)
        idx_real = jax.random.choice(sk_real, N_real, shape=(batch_size,), replace=False)
        idx_pair = jax.random.choice(sk_pair, N_pair, shape=(batch_size,), replace=False)

        x_s = {mod: dataset['sim'][mod][idx_sim] for mod in modalities}
        theta = dataset['sim'][sim_theta_key][idx_sim]
        x_t = {mod: dataset['real'][mod][idx_real] for mod in modalities}
        x_s_pair = {mod: dataset['pair'][mod][0][idx_pair] for mod in modalities}
        x_t_pair = {mod: dataset['pair'][mod][1][idx_pair] for mod in modalities}

        batch = {
            'source': (x_s, theta),
            'target': x_t,
            'x_s_pair': x_s_pair,
            'x_t_pair': x_t_pair
        }

        return batch, rng

    return get_batch

