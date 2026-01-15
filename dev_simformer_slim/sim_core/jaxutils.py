from typing import Callable, Tuple
from jaxtyping import PyTree, Array
from jax._src.flatten_util import ravel_pytree
# try the old public API, fall back to the internal core
try:
    from jax import linear_util as lu
except ImportError:
    import jax._src.linear_util as lu


    
@lu.transformation
def ravel_arg_(unravel, index, *args):
    flat_arg_i = args[index]
    arg_i = unravel(flat_arg_i)
    args = args[:index] + (arg_i,) + args[index+1:]
    ans = yield args, {}
    ans_flat, _ = ravel_pytree(ans)
    yield ans_flat      


def ravel_args(in_vals: PyTree) -> Tuple[Array, Callable]:
    """_summary_

    Args:
        in_vals (PyTree): _description_

    Returns:
        Tuple[Array, Callable]: _description_
    """
    flat_vals, unflatten = ravel_pytree(in_vals)
    return flat_vals, unflatten

def ravel_arg_fun(fun: Callable, unravel, index: int) -> Callable:
    return ravel_arg_(lu.wrap_init(fun), unravel, index).call_wrapped
