import jax
from jax import numpy as jnp
from jaxtyping import Array, Float
from typing import Callable


def linear_interpolation(ts: Array, ys: Array) -> Callable[[Float], Array]:
    """Linear interpolation function for a given set of points (ts, ys). Here ts must be a one dimensional sorted array and ys can be any array with the same length as ts on axis 0.
        Outside of the data range, the function returns the value of the nearest data point.

    Args:
        ts (Array): Time points
        ys (Array): Values at time points

    Returns:
        Callable[[Float], Array]: Interpolation function that can be evaluated at any time point.
    """

    shape = ys.shape
    event_shape = ys.shape[1:]
    ys = ys.reshape(shape[0], -1)

    def interpolate(t: Float) -> Array:
        return jax.vmap(jnp.interp, in_axes=(None, None, -1))(t, ts, ys).reshape(
            event_shape
        )

    return interpolate
