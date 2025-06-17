from functools import update_wrapper
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import core
from jax.extend.core import Primitive
from jax.interpreters import ad, batching, mlir
import jax._src.linear_util as lu
from jax._src.util import cache, safe_map
from jax._src.api_util import flatten_fun_nokwargs, argnums_partial, debug_info
from jax.tree_util import tree_flatten, tree_unflatten
from jax._src.interpreters.partial_eval import trace_to_jaxpr_dynamic

# Fallback for shaped_abstractify if the API changed
try:
    from jax._src.api_util import shaped_abstractify
except ImportError:
    def shaped_abstractify(x):
        return core.ShapedArray(jnp.shape(x), jnp.result_type(x))

# Configure print options
jax.config.update('jax_numpy_rank_promotion', 'warn')
jnp.set_printoptions(precision=3, suppress=True)

# Custom primitive for inverse calls
custom_inverse_call_p = Primitive("custom_inverse_call_p")
custom_inverse_call_p.multiple_results = True

@custom_inverse_call_p.def_impl
def custom_inverse_call_impl(*args, forward_jaxpr, inverse_jaxpr, **params):
    with core.new_sublevel():
        return core.eval_jaxpr(forward_jaxpr.jaxpr, forward_jaxpr.consts, *args)

@custom_inverse_call_p.def_abstract_eval
def custom_inverse_call_abstract_eval(*args, forward_jaxpr, inverse_jaxpr, **params):
    return forward_jaxpr.out_avals

# Lowering rule
def custom_inverse_call_lowering(ctx, *args, forward_jaxpr, inverse_jaxpr, **params):
    return mlir.call_lowering(
        ctx, *args, name="forward_call", call_jaxpr=forward_jaxpr
    )
mlir.register_lowering(custom_inverse_call_p, custom_inverse_call_lowering)

# JVP support
@cache()
def _process_jvp(fwd_jaxpr, tangents):
    nonzeros = [type(t) is not ad.Zero for t in tangents]
    jvp_jaxpr, _ = ad.jvp_jaxpr(fwd_jaxpr, nonzeros, instantiate=False)
    nz_t = [t for t in tangents if type(t) is not ad.Zero]
    return jvp_jaxpr, nz_t

# Register JVP rule

def custom_inverse_jvp(primals, tangents, forward_jaxpr, inverse_jaxpr, **params):
    jvp_jaxpr, nz_t = _process_jvp(forward_jaxpr, tangents)
    new_p, new_t = core.eval_jaxpr(
        jvp_jaxpr.jaxpr, jvp_jaxpr.consts, *primals, *nz_t
    )
    return [new_p], [new_t]

# Associate the JVP rule with the primitive
ad.primitive_jvps[custom_inverse_call_p] = custom_inverse_jvp

# Batching support
def batch_custom_inverse_call(
    spmd_axis_name, axis_size, axis_name, main_type, args, dims, **params
):
    fwd_jaxpr = params.pop("forward_jaxpr")
    inv_jaxpr = params.pop("inverse_jaxpr")
    batched_args = [batching.bdim_at_front(x, d, axis_size) for x, d in zip(args, dims)]
    batched_fwd, out_size = batching.batch_jaxpr(
        fwd_jaxpr, axis_size,
        [True] * len(fwd_jaxpr.in_avals), [True] * len(fwd_jaxpr.out_avals),
        axis_name, spmd_axis_name, main_type
    )
    batched_inv, _ = batching.batch_jaxpr(
        inv_jaxpr, axis_size,
        [True] * len(inv_jaxpr.in_avals), [True] * len(inv_jaxpr.out_avals),
        axis_name, spmd_axis_name, main_type
    )
    out = custom_inverse_call_p.bind(
        *batched_args,
        forward_jaxpr=batched_fwd,
        inverse_jaxpr=batched_inv,
    )
    dims_out = [0 if b else batching.not_mapped for b in out_size]
    return out, dims_out

# Register batching rules for vmap and xmap

def _batch_custom_inverse_call_vmap(args, dims, axis_name, main_type, **params):
    return batch_custom_inverse_call(None, None, axis_name, main_type, args, dims, **params)

batching.primitive_batchers[custom_inverse_call_p] = _batch_custom_inverse_call_vmap
try:
    batching.spmd_primitive_batchers[custom_inverse_call_p] = batch_custom_inverse_call
except AttributeError:
    pass

# Transpose support
ad.primitive_transposes[custom_inverse_call_p] = ad.call_transpose

# Tracing forward and inverse to Jaxprs
@cache()
def _trace_fwd_inv(
    fwd_fun, inv_fun, dyn_idx, inv_argnum, in_avals, in_tree, name
):
    f_flat, out_tree = flatten_fun_nokwargs(fwd_fun, in_tree)
    dbg = debug_info(fwd_fun, in_tree, out_tree, False, name)
    jaxpr, out_avals, consts = trace_to_jaxpr_dynamic(f_flat, in_avals, dbg)
    fwd_closed = core.ClosedJaxpr(jaxpr, consts)

    inv_flat, _ = flatten_fun_nokwargs(inv_fun, in_tree)
    inv_avals = list(in_avals)
    inv_avals[dyn_idx.index(inv_argnum)] = out_avals[0]
    jaxpr2, _, consts2 = trace_to_jaxpr_dynamic(inv_flat, tuple(inv_avals), dbg)
    inv_closed = core.ClosedJaxpr(jaxpr2, consts2)
    return fwd_closed, inv_closed, out_tree()

class custom_inverse:
    def __init__(self, fun: Callable, inv_argnum=0, static_argnums=None):
        update_wrapper(self, fun)
        self.fun = fun
        self.inv_argnum = inv_argnum
        self.static_argnums = static_argnums
        self.inv_fun = None
        self.inv_fun_logdet = None

    def definv(self, inv: Callable) -> Callable:
        self.inv_fun = inv
        def wrapped(*args, **kwargs):
            return inv(*args, **kwargs), jnp.nan
        self.inv_fun_logdet = wrapped
        return wrapped

    def definv_and_logdet(self, fn: Callable) -> Callable:
        self.inv_fun_logdet = fn
        if self.inv_fun is None:
            self.inv_fun = lambda *a, **k: fn(*a, **k)[0]
        return fn

    def __call__(self, *args, **params) -> Any:
        if self.inv_fun is None:
            raise AttributeError(f"No inverse defined for {self.fun.__name__}")
        name = self.fun.__name__
        # Handle static args
        if self.static_argnums is None:
            f_dyn = lu.wrap_init(self.fun, params=params)
            inv_dyn = lu.wrap_init(self.inv_fun_logdet, params=params)
            dyn_idx = list(range(len(args)))
            dyn_args = args
        else:
            dyn_idx = [i for i in range(len(args)) if i not in self.static_argnums]
            f_dyn, dyn_args = argnums_partial(self.fun, dyn_idx, args)
            inv_dyn, _ = argnums_partial(self.inv_fun_logdet, dyn_idx, args)
        args_flat, in_tree = tree_flatten(dyn_args)
        avals = tuple(safe_map(shaped_abstractify, args_flat))
        fwd_jaxpr, inv_jaxpr, out_tree = _trace_fwd_inv(
            f_dyn, inv_dyn, dyn_idx, self.inv_argnum, avals, in_tree, name
        )
        out_flat = custom_inverse_call_p.bind(
            *args_flat,
            forward_jaxpr=fwd_jaxpr,
            inverse_jaxpr=inv_jaxpr,
        )
        return tree_unflatten(out_tree, out_flat)
