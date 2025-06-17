import jax
import jax.numpy as jnp
from jax import random as jrandom
from functools import partial
from sim_core.sdeint import sdeint
from model_trainer import TrainScoreModel
from data_handling import denormalize_data



class SimFormer(TrainScoreModel):
    def __init__(self, time_steps, **kwargs):
        super().__init__(**kwargs)
        self.end_std = jnp.squeeze(self.sde.marginal_stddev(jnp.ones(1)))
        self.end_mean = jnp.squeeze(self.sde.marginal_mean(jnp.ones(1)))
        self.time_steps = time_steps

    def set_model_params(self, params):
        self.opt_params = params

    def drift_backward(self, t, x, condition_mask, edge_mask, node_ids=None, replace_conditioned=True):
        if node_ids is None:
            node_ids = self.node_ids

        if self.opt_params is None:
            raise ValueError('Model has not yet been trained')
        score = self.model_fn(
            self.opt_params,
            t.reshape(-1, 1, 1),
            # t.reshape(-1, 1),
            x.reshape(-1, len(node_ids), 1),
            node_ids,
            condition_mask[:len(node_ids)],
            edge_mask=edge_mask
        )
        score = score.reshape(x.shape)
        # jax.debug.print("score dif is {}",score )
        f = self.sde.drift(t, x) - self.sde.diffusion(t,x)**2 * score
        if replace_conditioned:
            f = f * (1-condition_mask[:len(node_ids)])
        return f

    # Reverse SDE diffusion
    def diffusion_backward(self, t, x, condition_mask, node_ids=None, replace_conditioned=True):
        if node_ids is None:
            node_ids = self.node_ids

        b = self.sde.diffusion(t, x)
        # jax.debug.print("diffusion is {}",b )
        if replace_conditioned:
            b = b * (1-condition_mask[:len(node_ids)])
        return b

    @partial(jax.jit, static_argnums=(0, 2, 6))
    def sample_fn(self, key, shape, condition_mask, condition_value, node_ids=None, edge_mask=None, replace_conditioned=True):
        if node_ids is None:
            node_ids = self.node_ids

        condition_mask = condition_mask[:len(node_ids)]
        key1, key2 = jrandom.split(key, 2)
        x_T = jax.random.normal(key1, shape + (len(node_ids),)) * self.end_std[node_ids] + self.end_mean[node_ids]

        if replace_conditioned:
            x_T = x_T * (1-condition_mask) + condition_value * condition_mask

        keys = jrandom.split(key2, shape)
        ys = jax.vmap(
            lambda *args: sdeint(*args, noise_type="diagonal"), in_axes=(0, None, None, 0, None), out_axes=0
        )(
            keys,
            lambda t, x: self.drift_backward(
                t, x, condition_mask, node_ids=node_ids, edge_mask=edge_mask, replace_conditioned=replace_conditioned),
            lambda t, x: self.diffusion_backward(
                t, x, condition_mask, node_ids=node_ids, replace_conditioned=replace_conditioned),
            x_T, jnp.linspace(1., self.T_min, self.time_steps)
        )
        return ys

    def sample(self, key, shape, condition_mask, condition_value, edge_mask=None, replace_conditioned=True):
        """
        Sample from the model using a reverse SDE diffusion process.

        Args:
            key: Random key for sampling.
            shape: Shape of the samples to generate.
            condition_mask: Mask indicating which nodes are conditioned.
            condition_value: Values for the conditioned nodes.
            edge_mask: Optional mask for edges (default is None).
            replace_conditioned: Whether to replace conditioned values (default is True).

        Returns:
            Samples generated from the model.
        """
        ys = self.sample_fn(key, shape, condition_mask, condition_value, edge_mask, replace_conditioned)
        # Denormalize the samples
        return denormalize_data(ys, self.means, self.stds)