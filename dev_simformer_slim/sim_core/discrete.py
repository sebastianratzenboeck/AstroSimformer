import jax.numpy as jnp
from jax import random

from jaxtyping import Array

from .distribution import Distribution
from .constraints import (
    finit_set,
    simplex,
    real,
)

__all__ = [
    "Empirical",
]

from jax.tree_util import register_pytree_node_class


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
            return jnp.mean(self.values, axis=0)
        else:
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
            return jnp.var(self.values, axis=0)
        else:
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
