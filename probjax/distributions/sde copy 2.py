import jax
import jax.numpy as jnp
from jax.random import PRNGKeyArray

from functools import partial
from typing import Callable, Union, Optional
from jaxtyping import Array

from probjax.distributions import Distribution
from probjax.utils.sdeint import sdeint
from probjax.utils.odeint import odeint


### Base Class for SDEs ###
class BaseSDE:
    def __init__(
        self,
        drift: Callable[[Array, Array], Array],
        diffusion: Callable[[Array, Array], Array],
        p0: Distribution,
    ):
        self.drift = drift
        self.diffusion = diffusion
        self.p0 = p0

    def drift_fn(self, t: Array, x: Array) -> Array:
        return self.drift(t, x)

    def diffusion_fn(self, t: Array, x: Array) -> Array:
        return self.diffusion(t, x)

    def sample_initial(self, key: PRNGKeyArray, sample_shape=()) -> Array:
        return self.p0.sample(seed=key, sample_shape=sample_shape)

    def log_prob_initial(self, x0: Array) -> Array:
        return self.p0.log_prob(x0)


### Linear Time-Variant SDE ###
class LinearTimeVariantSDE(BaseSDE):
    def __init__(
        self,
        drift_matrix: Callable[[Array], Array],
        diffusion_matrix: Callable[[Array], Array],
        p0: Distribution,
    ):
        # Wrap matrices to make them compatible with the general SDE API
        drift = lambda t, x: jnp.einsum("...ij,...j->...i", drift_matrix(t), x)
        diffusion = lambda t, x: diffusion_matrix(t) * jnp.ones_like(x)
        super().__init__(drift, diffusion, p0)

        self.drift_matrix = drift_matrix
        self.diffusion_matrix = diffusion_matrix

    def sample_initial(self, key: PRNGKeyArray, sample_shape=()) -> Array:
        return super().sample_initial(key, sample_shape=sample_shape)

    def log_prob_initial(self, x0: Array) -> Array:
        return super().log_prob_initial(x0)


### Variance-Exploding SDE ###
class VESDE(LinearTimeVariantSDE):
    def __init__(
        self,
        p0: Distribution,
        sigma_max: Union[Array, float] = 10.0,
        sigma_min: Union[Array, float] = 0.01,
    ) -> None:
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min

        shape = p0.event_shape
        d = shape[0] if len(shape) > 0 else 1
        _const = jnp.sqrt(2 * jnp.log(sigma_max / sigma_min))

        # Drift matrix remains unchanged
        drift_matrix = lambda t: jnp.zeros(1)

        # Diffusion matrix handles both `t` and `x`
        diffusion_matrix = lambda t, x=None: jnp.atleast_1d(
            sigma_min * (sigma_max / sigma_min) ** t * _const
        )

        # Pass only `t` to diffusion_matrix by wrapping it in the parent class
        super().__init__(
            drift_matrix=drift_matrix,
            diffusion_matrix=lambda t: diffusion_matrix(t, None),
            p0=p0,
        )

    def marginal_mean(self, ts: Array, x0=None, **kwargs) -> Array:
        if x0 is None:
            # Use nanmean to ignore NaN values in self.p0.mean
            mu0 = jnp.nanmean(self.p0.mean)  # Adjusted to handle NaNs
        else:
            # Mask NaNs in x0 to ensure they don't affect the result
            mu0 = jnp.nan_to_num(x0, nan=0.0)  # Replace NaNs with 0 for consistency

        while ts.ndim < mu0.ndim:
            ts = jnp.expand_dims(ts, axis=-1)

        ts, mu0 = jnp.broadcast_arrays(ts, mu0)

        return mu0

    def mean(self, ts: Array, x0=None, **kwargs) -> Array:
        shape = ts.shape
        ts = jnp.expand_dims(
            ts,
            axis=(
                -i for i in range(1, len(self.batch_shape) + len(self.event_shape) + 1)
            ),
        )
        mu = self.marginal_mean(ts)
        return mu.reshape(shape + self.batch_shape + self.event_shape)

    def variance(self, ts: Array, x0=None, **kwargs) -> Array:
        shape = ts.shape
        ts = jnp.expand_dims(
            ts,
            axis=(
                -i for i in range(1, len(self.batch_shape) + len(self.event_shape) + 1)
            ),
        )
        var = self.marginal_variance(ts)
        return var.reshape(shape + self.batch_shape + self.event_shape)

    def marginal_variance(self, ts: Array, x0=None, **kwargs) -> Array:
        if x0 is None:
            var0 = self.p0.variance
        else:
            var0 = jnp.zeros_like(x0)

        while ts.ndim < var0.ndim:
            ts = jnp.expand_dims(ts, axis=-1)

        ts, var0 = jnp.broadcast_arrays(ts, var0)

        vart = self.sigma_min**2 * (self.sigma_max / self.sigma_min) ** (2 * ts)
        var = var0 + vart
        return var
    def marginal_stddev(self, t: Array, x0=None, **kwargs) -> Array:
        return jnp.sqrt(self.marginal_variance(t, x0, **kwargs))

    def sample_marginal(
        self, key: PRNGKeyArray, t: Array, sample_shape=(), x0=None, **kwargs
    ) -> Array:
        mean = self.mean(t, x0)
        std = self.stddev(t, x0)
        eps = jax.random.normal(key, sample_shape + mean.shape)
        return mean + std * eps
