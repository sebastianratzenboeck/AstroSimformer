import jax
import jax.numpy as jnp
from jax.random import PRNGKeyArray

from functools import partial
from typing import Callable, Union, Optional
from jaxtyping import Array

from probjax.distributions import Distribution, Independent, Normal
from probjax.distributions.discrete import Empirical
from probjax.utils.linalg import (
    is_matrix,
    is_diagonal_matrix,
    transition_matrix,
    matrix_fraction_decomposition,
)
from probjax.utils.sdeint import sdeint
from probjax.utils.odeint import odeint


class BaseSDE(Distribution):
    noise_type: str = "general"

    def __init__(self, drift: Callable, diffusion: Callable, p0: Distribution) -> None:
        """A base class for SDEs. We assume that the SDE is of the form:

        dX_t = f(t, X_t)dt + g(t, X_t)dW_t

        where f and g are the drift and diffusion functions respectively. We assume that the initial distribution is given by p0 at time t=0.

        Args:
            drift (Callable): Drift function
            diffusion (Callable): Diffusion function
            p0 (Distribution): Initial distribution
        """
        self.drift = drift
        self.diffusion = diffusion
        self.p0 = p0

        super().__init__(batch_shape=p0.batch_shape, event_shape=p0.event_shape)

    def mean(self, t: Array) -> Array:
        assert jnp.all(t >= 0), "t must be positive"
        # TODO: Implement through linearization
        raise NotImplementedError

    def marginal_mean(self, t: Array, x0=None, **kwargs) -> Array:
        assert jnp.all(t >= 0), "t must be positive"
        raise NotImplementedError

    def variance(self, t: Array) -> Array:
        assert jnp.all(t >= 0), "t must be positive"
        # TODO: Implement through linearization
        raise NotImplementedError

    def marginal_variance(self, t: Array, x0=None, **kwargs) -> Array:
        assert jnp.all(t >= 0), "t must be positive"
        raise NotImplementedError

    def stddev(self, t: Array) -> Array:
        return jnp.sqrt(self.variance(t))

    def marginal_stddev(self, t: Array, x0=None, **kwargs) -> Array:
        return jnp.sqrt(self.marginal_variance(t, x0, **kwargs))

    def covariance_matrix(self, t: Array) -> Array:
        assert jnp.all(t >= 0), "t must be positive"
        raise NotImplementedError

    def marginal_covariance_matrix(self, t: Array, x0=None, **kwargs) -> Array:
        assert jnp.all(t >= 0), "t must be positive"
        raise NotImplementedError

    def cross_covariance(self, t1: Array, t2: Array, x0=None, **kwargs) -> Array:
        assert jnp.all(t1 >= 0), "t1 must be positive"
        assert jnp.all(t2 >= 0), "t2 must be positive"
        raise NotImplementedError

    def marginal_cross_covariance(
        self, t1: Array, t2: Array, x0=None, **kwargs
    ) -> Array:
        assert jnp.all(t1 >= 0), "t1 must be positive"
        assert jnp.all(t2 >= 0), "t2 must be positive"
        raise NotImplementedError

    def cross_covariance_matrix(self, t1: Array, t2: Array, x0=None, **kwargs) -> Array:
        assert jnp.all(t1 >= 0), "t1 must be positive"
        assert jnp.all(t2 >= 0), "t2 must be positive"
        raise NotImplementedError

    def marginal_cross_covariance_matrix(
        self, t1: Array, t2: Array, x0=None, **kwargs
    ) -> Array:
        assert jnp.all(t1 >= 0), "t1 must be positive"
        assert jnp.all(t2 >= 0), "t2 must be positive"
        raise NotImplementedError

    def marginal_rsample(
        self, key: PRNGKeyArray, t: Array, sample_shape=(), x0=None, **kwargs
    ) -> Array:
        raise NotImplementedError

    def marginal_sample(self, key: PRNGKeyArray, t: Array, sample_shape=(), **kwargs):
        return self.marginal_rsample(key, t, sample_shape, **kwargs)

    def rsample(self, key: PRNGKeyArray, ts: Array, sample_shape=(), **kwargs) -> Array:
        """Samples from the SDE

        Args:
            key (PRNGKeyArray): Random key
            ts (Array): Number of time points to evaluate the SDE
            sample_shape (tuple, optional): Number of samples. Defaults to ().
            **kwargs: Additional arguments to pass to the solver i.e. see sdeint in probjax/utils/sdeint.py for more details

        Returns:
            Array: Samples from the SDE of shape (sample_shape, batch_shape, event_shape)
        """
        assert jnp.all(ts >= 0), "t must be positive"
        key1, key2 = jax.random.split(key)

        # Sample initial values
        x0 = self.p0.sample(key1, sample_shape)

        # Flatten and split keys
        x0_flat = x0.reshape(-1, *self.event_shape)
        keys_flat = jax.random.split(key2, x0_flat.shape[0])
        if ts.ndim <= 1:
            vmap_dim = None
        else:
            ts = ts.reshape(-1, ts.shape[-1])
            vmap_dim = 0

        # Sdeint
        _sdeint = partial(sdeint, **kwargs)
        __sdeint = jax.vmap(_sdeint, in_axes=(0, None, None, 0, vmap_dim))
        ys = __sdeint(keys_flat, self.drift, self.diffusion, x0_flat, ts)

        # Reshape to correct shape
        ys = ys.reshape(sample_shape + self.batch_shape + ts.shape + self.event_shape)
        return ys

    def sample(self, key: PRNGKeyArray, ts: Array, sample_shape=(), **kwargs) -> Array:
        return self.rsample(key, ts, sample_shape, **kwargs)

    def log_prob(self, x: Array, t: Array) -> Array:
        assert jnp.all(t >= 0), "t must be positive"
        raise NotImplementedError


# TODO This must be refactored


class LinearTimeInvariantSDE(BaseSDE):
    noise_type: str = "general"

    def __init__(
        self,
        drift_matrix: Array,
        diffusion_matrix: Array,
        p0: Distribution,
    ) -> None:
        """This class represents a linear time invariant SDE of the form:

        dX_t = A X_t dt + B dW_t

        where A and B are matrices and W_t is a Wiener process. The initial distribution is given by p0 at time t=0.

        Args:
            drift_matrix (Array): The drift matrix A
            diffusion_matrix (Array): The diffusion matrix B
            p0 (Distribution): The initial distribution
        """

        batch_shape = p0.batch_shape
        drift_matrix_format = drift_matrix[len(batch_shape) :].ndim
        diffusion_matrix_format = diffusion_matrix[len(batch_shape) :].ndim

        assert (
            drift_matrix_format <= 2 or drift_matrix.shape[1] == p0.event_shape[0]
        ), "Drift matrix must be compatible with initial distribution"
        assert (
            drift_matrix_format <= 2 or diffusion_matrix.shape[0] == p0.event_shape[0]
        ), "Diffusion matrix must be compatible with initial distribution"

        def drift(t, x):
            if drift_matrix_format == 1:
                return drift_matrix * x
            elif drift_matrix_format == 2:
                return jnp.matmul(drift_matrix, x)

        diffusion = lambda t, x: diffusion_matrix
        if diffusion_matrix_format == 2:
            self.noise_type = "general"
        else:
            self.noise_type = "diagonal"

        super().__init__(drift, diffusion, p0)

        # Store the matrices
        self.diffusion_matrix = diffusion_matrix
        self.drift_matrix = drift_matrix

    def mean(self, t: Array, x0=None, **kwargs) -> Array:
        assert jnp.all(t >= 0), "t must be positive"
        mu0 = self.p0.mean
        t = jnp.atleast_1d(t)

        P = jax.vmap(transition_matrix, in_axes=(None, None, 0))(
            self.drift_matrix, 0.0, t
        )

        if P.ndim == 3:
            return jnp.einsum("...ij,...j->...i", P, mu0)
        else:
            return P * mu0

    def covariance_matrix(self, t: Array, x0=None, **kwargs) -> Array:
        assert jnp.all(t >= 0), "t must be positive"
        assert (
            self.p0.event_shape != ()
        ), "Initial distribution must not be scalar, use var instead"
        Phi, Q = jax.vmap(matrix_fraction_decomposition, in_axes=(0, None, None, None))(
            t, 0.0, self.drift_matrix, self.diffusion_matrix
        )

        cov0 = self.p0.covariance_matrix

        return jnp.matmul(Phi, jnp.matmul(cov0, Phi.T)) + Q

    def variance(self, t: Array) -> Array:
        assert jnp.all(t >= 0), "t must be positive"
        t = jnp.atleast_1d(t)
        Phi, Q = jax.vmap(matrix_fraction_decomposition, in_axes=(None, 0, None, None))(
            0.0, t, self.drift_matrix, self.diffusion_matrix
        )
        var0 = self.p0.variance
        var = Phi**2 * var0 + Q
        return jnp.squeeze(var, axis=-1)

    def sample_marginal(
        self, key: PRNGKeyArray, t: Array, sample_shape=(), x0=None, **kwargs
    ) -> Array:
        mean = self.mean(t, x0)
        cov = self.covariance_matrix(t, x0)
        L = jnp.linalg.cholesky(cov)

        eps = jax.random.normal(key, sample_shape + mean.shape)
        return mean + jnp.matmul(L, eps[..., None])[..., 0]


class LinearTimeVariantSDE(BaseSDE):
    def __init__(
        self, drift_matrix: Callable, diffusion_matrix: Callable, p0: Distribution
    ) -> None:
        self.drift_matrix = drift_matrix
        self.diffusion_matrix = diffusion_matrix

        batch_shape = p0.batch_shape
        drift_matrix_format = drift_matrix(0)[len(batch_shape) :].ndim

        def drift(t, x):
            if drift_matrix_format == 1:
                return drift_matrix(t) * x
            elif drift_matrix_format == 2:
                return jnp.matmul(drift_matrix(t), x)

        def diffusion(t, x):
            B = diffusion_matrix(t)
            if B.ndim == 1:
                return B * jnp.ones_like(x)
            else:
                return B

        super().__init__(drift, diffusion, p0)

    def mean(self, ts: Array, **kwargs) -> Array:
        assert jnp.all(ts >= 0), "t must be positive"

        _odeint = partial(odeint, **kwargs)
        if self.batch_shape != ():
            _odeint = jax.vmap(_odeint, in_axes=(None, 0, None))
        mu0 = self.p0.mean
        mus = _odeint(self.drift, mu0, ts)

        return mus

    def variance(self, t: Array, **kwargs) -> Array:
        assert jnp.all(t >= 0), "t must be positive"
        if self.p0.event_shape != ():
            cov = self.covariance_matrix(t, **kwargs)
            return jnp.sum(jnp.diagonal(cov, axis1=-2, axis2=-1))
        else:
            var0 = self.p0.variance
            _odeint = partial(odeint, **kwargs)
            if self.batch_shape != ():
                _odeint = jax.vmap(_odeint, in_axes=(None, 0, None))

            def f(t, var):
                return self.drift_matrix(t) ** 2 * var + self.diffusion_matrix(t) ** 2

            vars = _odeint(f, var0, t)
            return vars

    def covariance_matrix(self, t: Array, **kwargs) -> Array:
        assert jnp.all(t >= 0), "t must be positive"
        assert (
            self.p0.event_shape != ()
        ), "Initial distribution must not be scalar, use var instead"
        _odeint = partial(odeint, **kwargs)
        if self.batch_shape != ():
            _odeint = jax.vmap(_odeint, in_axes=(None, 0, None))
        cov0 = self.p0.covariance_matrix

        def f(t, cov):
            term1 = jnp.matmul(self.drift(t), cov)
            term2 = jnp.matmul(cov, self.drift(t).T)
            term3 = jnp.matmul(self.diffusion(t), self.diffusion(t).T)
            return term1 + term2 + term3

        covs = _odeint(f, cov0, t)
        return covs

    def log_prob(self, x: Array, t: Array, x0=None) -> Array:
        mu = self.mean(t, x0=x0)
        std = self.std(t, x0=x0)

        return jax.scipy.stats.norm.logpdf(x, mu, std)



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
        drift_matrix = lambda t: jnp.zeros(1)
        diffusion_matrix = lambda t: jnp.atleast_1d(
            sigma_min * (sigma_max / sigma_min) ** t * _const
        )

        super().__init__(drift_matrix, diffusion_matrix, p0)
        
    import jax.numpy as jnp

    def marginal_mean(self, ts: Array, x0=None, **kwargs) -> Array:

        if x0 is None:
            mu0 = self.p0.mean
        else:
            # Calculate the nanmean along axis 0 (ignoring NaNs) for each feature in x0
            mu0 = jnp.nanmean(x0, axis=0)  # Calculate the mean for each feature, ignoring NaNs
            
            # Replace NaNs in x0 with the computed mean
            x0 = jnp.nan_to_num(x0, nan=mu0)
        
        # Expand dimensions of ts to match mu0 if necessary
        while ts.ndim < mu0.ndim:
            ts = jnp.expand_dims(ts, axis=-1)
        
        # Broadcast arrays to ensure they match in shape
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

    def sample_marginal(
        self, key: PRNGKeyArray, t: Array, sample_shape=(), x0=None, **kwargs
    ) -> Array:
        mean = self.mean(t, x0)
        std = self.stddev(t, x0)
        eps = jax.random.normal(key, sample_shape + mean.shape)
        return mean + std * eps



import jax.numpy as jnp
from jax.scipy.special import logsumexp

import jax
from jax import random
from jax.lax import scan

from typing import Optional, Sequence, Union

from probjax.distributions.distribution import Distribution
from probjax.distributions.constraints import real, positive, unit_interval

__all__ = ["Independent"]

from jax.tree_util import register_pytree_node_class

# Transforms a batch of independent distributions into a single mulitvariate product distribution.

import jax.numpy as jnp
from jax.scipy.special import logsumexp

import jax
from jax import random
from jax.lax import scan

from typing import Optional, Sequence, Union


__all__ = ["Independent"]

from jax.tree_util import register_pytree_node_class

# Transforms a batch of independent distributions into a single mulitvariate product distribution.


@register_pytree_node_class
class Independent(Distribution):
    """
    Creates an independent distribution by treating the provided distribution as
    a batch of independent distributions.

    Args:
        base_dist: Base distribution object.
        reinterpreted_batch_ndims: The number of batch dimensions that should
            be considered as event dimensions.
    """

    def __init__(
        self,
        base_dist: Union[Distribution, Sequence[Distribution]],
        reinterpreted_batch_ndims: int,
    ):
        # Determine batch_shape and event_shape using the helper function
        batch_shape, event_shape, event_ndims, reinterpreted_batch_ndims = determine_shapes(
            base_dist, reinterpreted_batch_ndims
        )
        

        if isinstance(base_dist, Distribution):
            # Single distribution case
            self.base_dist = [base_dist]
        else:
            self.base_dist = base_dist

        self.event_ndims = event_ndims
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims
        
        for p in self.base_dist:
            p._batch_shape = batch_shape
            p._event_shape = event_shape

        super().__init__(batch_shape=batch_shape, event_shape=event_shape)

    @property
    def mean(self):
        return jnp.stack([b.mean for b in self.base_dist], axis=-1).reshape(
            self.batch_shape + self.event_shape
        )

    @property
    def median(self):
        return jnp.stack([b.median for b in self.base_dist], axis=-1).reshape(
            self.batch_shape + self.event_shape
        )

    @property
    def mode(self):
        # The mode does change and is not equal to the mode of the base distribution
        raise NotImplementedError()

    @property
    def variance(self):
        return jnp.stack([b.variance for b in self.base_dist], axis=-1).reshape(
            self.batch_shape + self.event_shape
        )

    def rsample(self, key, sample_shape=()):
        keys = random.split(key, len(self.base_dist))
        samples = jnp.stack(
            [p.rsample(k, sample_shape) for k, p in zip(keys, self.base_dist)],
            axis=-1,
        )
        return jnp.reshape(samples, sample_shape + self.batch_shape + self.event_shape)

    def sample(self, key, sample_shape=()):
        keys = random.split(key, len(self.base_dist))
        if self.reinterpreted_batch_ndims > 0:
            samples = jnp.hstack(
                [p.sample(k, sample_shape) for k, p in zip(keys, self.base_dist)],
            )
        else:
            samples = jnp.stack(
                [p.sample(k, sample_shape) for k, p in zip(keys, self.base_dist)],
                axis=-len(self.event_shape) - 1,
            )
        return jnp.reshape(samples, sample_shape + self.batch_shape + self.event_shape)

    def log_prob(self, value):
        if len(self.base_dist) == 1:
            log_prob = self.base_dist[0].log_prob(value)
        else:
            if self.reinterpreted_batch_ndims > 0:
                split_value = jnp.split(value,self.event_ndims, axis=-1)[1:]
                log_prob = jnp.stack(
                    [
                        b.log_prob(v.reshape((-1,) + b.batch_shape + b.event_shape))
                        for b, v in zip(self.base_dist, split_value)
                    ], axis=-1
                )
                log_prob = jnp.reshape(
                    log_prob, value.shape[:-1] + self.batch_shape + (len(self.base_dist),)
                )
            else:
                split_value = jnp.split(
                    value, self.event_ndims, axis=-len(self.event_shape) - 1
                )[1:]
                log_prob = jnp.stack(
                    [b.log_prob(v) for b, v in zip(self.base_dist, split_value)],
                    axis=-len(self.event_shape) - 1,
                )
                log_prob = jnp.reshape(
                    log_prob,
                    value.shape[: -len(self.event_shape) - 1] + self.batch_shape,
                )

        # Sum the log probabilities along the event dimensions
        if self.reinterpreted_batch_ndims > 0:
            return jnp.sum(
                log_prob, axis=tuple(range(-self.reinterpreted_batch_ndims, 0))
            )
        else:
            return log_prob

    def entropy(self):
        entropy = jnp.stack([b.entropy() for b in self.base_dist], axis=-1)

        # Sum the entropies along the event dimensions
        if self.reinterpreted_batch_ndims > 0:
            return jnp.sum(
                entropy, axis=tuple(range(-self.reinterpreted_batch_ndims, 0))
            )
        else:
            return entropy

    def __repr__(self) -> str:
        return f"Independent({self.base_dist}, reinterpreted_batch_ndims={self.reinterpreted_batch_ndims})"

        # Each distribution will be registered as a PyTree

    def tree_flatten(self):
        flat_components, tree_components = jax.tree_util.tree_flatten(self.base_dist)
        return (
            tuple(flat_components),
            [tree_components, self.reinterpreted_batch_ndims],
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        tree_components, reinterpreted_batch_ndims = aux_data
        return cls(
            jax.tree_util.tree_unflatten(tree_components, children),
            reinterpreted_batch_ndims,
        )


def determine_shapes(
    base_dist: Union[Distribution, Sequence[Distribution]],
    reinterpreted_batch_ndims: int,
):
    if isinstance(base_dist, Distribution):
        # Single distribution case
        base_dist = [base_dist]

    # Extract batch shapes and event shapes from the list of base distributions
    batch_shapes = [b.batch_shape for b in base_dist]
    event_shapes = [b.event_shape for b in base_dist]
    
    assert all(reinterpreted_batch_ndims <= len(b) for b in batch_shapes) or all(reinterpreted_batch_ndims <= len(e) for e in event_shapes), "reinterpreted_batch_ndims must be greater than or equal to the batch shape of the base distribution."

    # Ensure that batch shapes are equal and calculate event_shape
    batch_shape, event_shape, event_ndims = calculate_shapes(
        batch_shapes, event_shapes, reinterpreted_batch_ndims
    )

    return tuple(batch_shape), tuple(event_shape), tuple(event_ndims), reinterpreted_batch_ndims


def calculate_shapes(batch_shapes, event_shapes, reinterpreted_batch_ndims):
    event_shape = list(event_shapes[0])
    if reinterpreted_batch_ndims > 0:
        new_event_shape = list(batch_shapes[0][-reinterpreted_batch_ndims:])
        if len(new_event_shape) > 0:
            for b in batch_shapes[1:]:
                if len(b) > 0:
                    new_event_shape[-1] += b[- 1]
                else:
                    new_event_shape[-1] += 1

        if len(event_shape) > 0:
            for e in event_shapes[1:]:
                if len(e) > 0:
                    event_shape[-1] += e[-1]
                else:
                    event_shape[-1] += 1

        batch_shape = tuple(batch_shapes[0][:-reinterpreted_batch_ndims])
        event_shape = tuple(new_event_shape) + tuple(event_shape)
        event_ndims = [0]
        for e in event_shapes:
            if len(e) == 0:
                event_ndims.append(event_ndims[-1] + 1)
            else:
                event_ndims.append(event_ndims[-1] + e[-1])
    else:
        new_batch_shape = list(batch_shapes[0])
        if len(new_batch_shape) > 0:
            for b in batch_shapes[1:]:
                if len(b) > 0:
                    new_batch_shape[-1] += b[-1]
                else:
                    new_batch_shape[-1] += 1
        else:
            new_batch_shape = (len(batch_shapes),)
        batch_shape = tuple(new_batch_shape)
        event_shape = tuple(event_shape)
        event_ndims = [0]
        for b in batch_shapes:
            if len(b) == 0:
                event_ndims.append(event_ndims[-1] + 1)
            else:
                event_ndims.append(event_ndims[-1] + b[-1])

    return batch_shape, event_shape, event_ndims

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax import lax
from jax.scipy.special import erfinv, erf

from jaxtyping import Array
from probjax.distributions.exponential_family import ExponentialFamily
from probjax.distributions.constraints import (
    finit_set,
    simplex,
    real,
    unit_interval,
    unit_integer_interval,
    positive_integer,
    strict_positive_integer,
)
import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax import lax
from jax.scipy.special import erfinv, erf

from jaxtyping import Array

from jax.scipy.stats import bernoulli, binom, poisson, geom, multinomial

__all__ = [
    "Empirical",
    "Dirac",
    "Bernoulli",
    "Binomial",
    "Poisson",
    "Geometric",
    "Categorical",
]

from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class Bernoulli(ExponentialFamily):
    arg_constraints = {"probs": unit_interval}
    support = unit_integer_interval

    def __init__(self, probs: Array):
        self.probs = jnp.asarray(probs)
        super().__init__(batch_shape=probs.shape, event_shape=())

    def sample(self, key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        return random.bernoulli(key, self.probs, shape=shape)

    def log_prob(self, value: Array) -> Array:
        return bernoulli.logpmf(value, self.probs)

    @property
    def mean(self) -> Array:
        return self.probs

    @property
    def variance(self) -> Array:
        return self.probs * (1 - self.probs)

    @property
    def entropy(self) -> Array:
        return (
            jnp.log(2)
            - self.probs * jnp.log(self.probs)
            - (1 - self.probs) * jnp.log(1 - self.probs)
        )

    def cdf(self, value: Array) -> Array:
        return bernoulli.cdf(value, self.probs)

    def icdf(self, value: Array) -> Array:
        return bernoulli.ppf(value, self.probs)


@register_pytree_node_class
class Binomial(ExponentialFamily):
    arg_constraints = {"n": strict_positive_integer, "probs": unit_interval}

    def __init__(self, n: Array, probs: Array):
        n, probs = jnp.broadcast_arrays(n, probs)

        self.n = n.astype(jnp.int32)
        self.probs = probs
        super().__init__(batch_shape=probs.shape, event_shape=())

    def sample(self, key, sample_shape=()):
        max_n = jnp.max(self.n)
        shape = sample_shape + (max_n,) + self.batch_shape + self.event_shape

        trials = random.bernoulli(key, self.probs, shape=shape)
        ax = -len(self.batch_shape) - len(self.event_shape) - 1
        sumed_trials = jnp.cumsum(trials, axis=ax)

        ns = jnp.expand_dims(self.n, axis=tuple(range(len(shape) - 1)))
        _take = jax.vmap(lambda x, y: jnp.take(x, y, axis=-1), in_axes=(-1, -1))
        final = _take(sumed_trials, ns - 1)
        final = jnp.transpose(final).reshape(
            sample_shape + self.batch_shape + self.event_shape
        )
        return final

    def log_prob(self, value: Array) -> Array:
        return binom.logpmf(value, self.n, self.probs)

    def cdf(self, value: Array) -> Array:
        return binom.cdf(value, self.n, self.probs)

    def icdf(self, value: Array) -> Array:
        return binom.ppf(value, self.n, self.probs)

    @property
    def mean(self) -> Array:
        return self.n * self.probs

    @property
    def median(self) -> Array:
        return jnp.floor(self.n * self.probs)

    @property
    def mode(self) -> Array:
        return jnp.floor((self.n + 1) * self.probs)

    @property
    def variance(self) -> Array:
        return self.n * self.probs * (1 - self.probs)

    @property
    def entropy(self) -> Array:
        return (
            jnp.log(2)
            - self.probs * jnp.log(self.probs)
            - (1 - self.probs) * jnp.log(1 - self.probs)
        )


@register_pytree_node_class
class Categorical(ExponentialFamily):
    arg_constraints = {"probs": simplex}

    def __init__(self, probs: Array):
        self.probs = jax.nn.softmax(probs)
        shape = self.probs.shape
        if len(shape) > 1:
            batch_shape = shape[:-1]
            event_shape = ()
        else:
            batch_shape = ()
            event_shape = ()

        super().__init__(batch_shape=batch_shape, event_shape=event_shape)

    def sample(self, key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        return random.categorical(key, self.probs, shape=shape, axis=-1)

    def log_prob(self, value: Array) -> Array:
        value = jnp.asarray(value).astype(jnp.int32)
        value = jax.nn.one_hot(value, self.probs.shape[-1])
        log_probs = jax.scipy.special.xlogy(value, self.probs).sum(axis=-1)
        return log_probs

    @property
    def mean(self) -> Array:
        return jnp.sum(self.probs * jnp.arange(self.probs.shape[-1]), axis=-1)

    @property
    def variance(self) -> Array:
        return jnp.sum(
            self.probs * (jnp.arange(self.probs.shape[-1]) - self.mean) ** 2, axis=-1
        )

    @property
    def entropy(self) -> Array:
        return -jnp.sum(self.probs * jnp.log(self.probs), axis=-1)


@register_pytree_node_class
class Poisson(ExponentialFamily):
    arg_constraints = {"rate": positive_integer}

    def __init__(self, rate: Array):
        self.rate = rate
        super().__init__(batch_shape=rate.shape, event_shape=())

    def sample(self, key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        return random.poisson(key, self.rate, shape=shape)

    def log_prob(self, value: Array) -> Array:
        return poisson.logpmf(value, self.rate)

    @property
    def mean(self) -> Array:
        return self.rate

    @property
    def variance(self) -> Array:
        return self.rate

    @property
    def entropy(self) -> Array:
        return self.rate * (1 - jnp.log(self.rate))

    def cdf(self, value: Array) -> Array:
        return poisson.cdf(value, self.rate)

    def icdf(self, value: Array) -> Array:
        return poisson.ppf(value, self.rate)


@register_pytree_node_class
class Geometric(ExponentialFamily):
    arg_constraints = {"probs": unit_interval}

    def __init__(self, probs: Array):
        self.probs = probs
        super().__init__(batch_shape=probs.shape, event_shape=())

    def sample(self, key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        return random.geometric(key, self.probs, shape=shape)

    def log_prob(self, value: Array) -> Array:
        return geom.logpmf(value, self.probs)

    @property
    def mean(self) -> Array:
        return 1 / self.probs

    @property
    def median(self) -> Array:
        return jnp.ceil(-jnp.log(2) / jnp.log(1 - self.probs))

    @property
    def variance(self) -> Array:
        return (1 - self.probs) / self.probs**2

    @property
    def entropy(self) -> Array:
        return -self.probs * jnp.log(self.probs) - (1 - self.probs) * jnp.log(
            1 - self.probs
        )

    def cdf(self, value: Array) -> Array:
        return geom.cdf(value, self.probs)

    def icdf(self, value: Array) -> Array:
        return geom.ppf(value, self.probs)


# Dirac delta distribution
@register_pytree_node_class
class Dirac(Distribution):
    arg_constraints = {"value": real}

    def __init__(self, value: Array):
        self.value = value
        self.support = finit_set(self.value)
        super().__init__(batch_shape=value.shape, event_shape=())

    def sample(self, key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        return jnp.broadcast_to(self.value, shape)

    def log_prob(self, value: Array) -> Array:
        true_value = jnp.broadcast_to(self.value, value.shape)
        return jnp.where(value == true_value, 0.0, -jnp.inf)

    @property
    def mean(self) -> Array:
        return self.value

    @property
    def variance(self) -> Array:
        return jnp.zeros(self.batch_shape)

    @property
    def entropy(self) -> Array:
        return jnp.zeros(self.batch_shape)

    def cdf(self, value: Array) -> Array:
        true_value = jnp.broadcast_to(self.value, value.shape)
        return jnp.where(value >= true_value, 1.0, 0.0)

    def icdf(self, value: Array) -> Array:
        true_value = jnp.broadcast_to(self.value, value.shape)
        return jnp.where(value >= 1.0, true_value, jnp.inf)


# Empirical distribution
@register_pytree_node_class
class Empirical(Distribution):
    arg_constraints = {"values": real, "probs": simplex}

    def __init__(self, values: Array, probs: Array | None = None):
        self.values = jnp.atleast_1d(values)
        self.support = finit_set(self.values)

        # Reinterpret the values as a batch of independent distributions
        self.num_values = self.values.shape[0]
        # Rest is interpreted as batch shape
        if values.ndim == 1:
            batch_shape = ()
            event_shape = ()
        else:
            batch_shape = self.values.shape[1:]
            event_shape = ()

        if probs is None:
            self.probs = None
        else:
            # assert probs.shape == values.shape, "probs shape mismatch"
            self.probs = jnp.atleast_1d(probs)


        super().__init__(batch_shape=batch_shape, event_shape=event_shape)

    def sample(self, key, sample_shape=()):
        shape = sample_shape + self.batch_shape
        base_index = jnp.arange(0, self.num_values)
        if self.probs is not None:
            base_index = jnp.broadcast_to(base_index, self.probs.shape)
        index = random.choice(
            key, base_index, shape=shape + (1,) * len(self._event_shape), p=self.probs
        )

        samples = jnp.take_along_axis(self.values, index, axis=0)
        return samples

    def log_prob(self, value: Array) -> Array:
        value = jnp.asarray(value)
        mask = jnp.equal(value[..., None], self.values)
        indices = jnp.argmax(mask, axis=-self.values.ndim)
        valid = jnp.any(mask, axis=-self.values.ndim)
        if self.probs is not None:
            probs = self.probs
            while probs.ndim < indices.ndim:
                probs = probs[None, ...]
            while indices.ndim < probs.ndim:
                indices = indices[None, ...]

            log_probs = jnp.take_along_axis(jnp.log(probs), indices, axis=-1)
            log_probs = jnp.where(valid, log_probs, -jnp.inf)
        else:
            log_probs = jnp.where(valid, -jnp.log(self.num_values), -jnp.inf)
        return log_probs

    @property
    def mean(self) -> Array:
        if self.probs is None:
            return jnp.nanmean(self.values, axis=0)
        else:
            raise NotImplementedError()
            return jnp.sum(self.values * self.probs, axis=0)

    @property
    def mode(self) -> Array:
        if self.probs is None:
            return jnp.bincount(self.values).argmax(axis=0)
        else:
            return self.values[jnp.argmax(self.probs)]

    @property
    def variance(self) -> Array:
        if self.probs is None:
            return jnp.nanvar(self.values, axis=0)
        else:
            raise NotImplementedError()
            return jnp.sum((self.values - self.mean) ** 2 * self.probs)

    @property
    def entropy(self) -> Array:
        if self.probs is None:
            return -jnp.log(self.num_values)
        else:
            return -jnp.sum(self.probs * jnp.log(self.probs), axis=0)

    def cdf(self, value: Array) -> Array:
        raise NotImplementedError()
        if self.probs is None:
            index = jnp.searchsorted(self.values, value)
            cumprobs = jnp.cumsum(self.probs)
            return cumprobs[index]
        else:
            return jnp.sum(self.probs * (self.values <= value[..., None]), axis=0)

    def icdf(self, value: Array) -> Array:
        if self.probs is None:
            return jnp.take_along_axis(self.values, value[..., None], axis=0)
        else:
            raise NotImplementedError()
