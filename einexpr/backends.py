from collections import namedtuple
from pyclbr import 
from typing import Union, Callable

from .base_typing import EinarrayLike
from .utils import pytree_mapreduce
from .einarray_utils import *
import einexpr


Dimension = str

backend_array_types = dict()
backend_dim_kwargs_to_resolve = ['dim', 'axis']

function_registry = []


def single_arg_elementwise_dispatch(func, args, kwargs):
    assert isinstance(args, (list, tuple))
    assert isinstance(kwargs, dict)
    assert isinstance(args[0], EinarrayLike)
    assert all(not isinstance(arg, EinarrayLike) for arg in args[1:])
    assert all(not isinstance(arg, EinarrayLike) for arg in kwargs.values())
    return einexpr.einarray(func(args[0].a, *args[1:], **kwargs), dims=args[0].dims, ambiguous_dims=args[0].ambiguous_dims)


def multi_arg_elementwise_dispatch(func, args, kwargs):
    assert isinstance(args, (list, tuple))
    assert isinstance(kwargs, dict)
    assert all(isinstance(arg, EinarrayLike) for arg in args)
    assert all(not isinstance(arg, EinarrayLike) for arg in kwargs.values())
    ambiguous_dims = einexpr.calculate_ambiguous_final_aligned_dims(*(arg.dims for arg in args))
    ambiguous_dims |= {dim for arg in args for dim in arg.ambiguous_dims}
    raw_aligned_arrays, out_dims = einexpr.align_arrays(*args, return_output_dims=True)
    return einexpr.einarray(func(*raw_aligned_arrays, **kwargs), dims=out_dims, ambiguous_dims=ambiguous_dims)


def multi_dim_reduction_dispatch(func, args, kwargs):
    # TODO:
    # - Check that axes are in the array
    # - raise error when there are duplicate axes, esp of different types (Dimension, int, and negative int)
    assert isinstance(args, (list, tuple))
    assert isinstance(kwargs, dict)
    assert isinstance(args[0], EinarrayLike)
    assert all(not isinstance(arg, EinarrayLike) for arg in args[1:]) # TODO: mapreduce over pytree
    assert all(not isinstance(arg, EinarrayLike) for arg in kwargs.values())
    if 'axis' in kwargs:
        axis = kwargs.pop('axis')
    elif 'dims' in kwargs:
        axis = kwargs.pop('dims')
    else:
        axis = None
    if axis is None:
        axis = list(range(len(args[0].dims)))
    elif isinstance(axis, (Dimension, int)):
        axis = [axis]
    axis = [args[0].dims.index(dim) if isinstance(dim, Dimension) else dim for dim in axis]
    assert all(isinstance(dim, int) for dim in axis)
    axis = [i if i >= 0 else i + len(args[0].dims) for i in axis]
    out_dims = [dim for i, dim in enumerate(args[0].dims) if i not in axis]
    ambiguous_out_dims = args[0].ambiguous_dims - {dim for i, dim in enumerate(args[0].dims) if i in axis}
    return einexpr.einarray(func(args[0].a, *args[1:], axis=tuple(axis), **kwargs), dims=out_dims, ambiguous_dims=ambiguous_out_dims)


def single_dim_reduction_dispatch(func, args, kwargs):
    if 'axis' in kwargs and isinstance(kwargs['axis'], (tuple, list)):
        raise ValueError("Multiple axes not supported")
    elif 'dims' in kwargs and isinstance(kwargs['dims'], (tuple, list)):
        raise ValueError("Multiple axes not supported")
    return multi_dim_reduction_dispatch(func, args, kwargs)


def concatenation_dispatch(func, args, kwargs):
    assert isinstance(args, (list, tuple))
    assert isinstance(kwargs, dict)
    assert isinstance(args[0], (list, tuple)) and all(isinstance(arg, EinarrayLike) for arg in args[0])
    assert all(not isinstance(arg, EinarrayLike) for arg in args[1:])
    if 'axis' in kwargs:
        axes = kwargs.pop('axis')
    elif 'dims' in kwargs:
        axes = kwargs.pop('dims')
    else:
        axes = 0
    if axes is None:
        raise NotImplementedError("Axis must be specified")
    if isinstance(axes, (Dimension, int)):
        axes = [axes] * len(args[0])
    if not isinstance(axes, (list, tuple)):
        raise ValueError("Axes must be a list or tuple")
    axes = list(axes)
    for i, (axis, arg) in enumerate(zip(axes, args[0])):
        # If the axis is an integer, convert it to a Dimension
        if isinstance(axis, int):
            axis = arg.dims[axis]
            axes[i] = axis
        elif not isinstance(axis, Dimension):
            raise ValueError(f"Invalid axis {axis}")
        assert axis not in arg.ambiguous_dims
    # Check that arrays share all dimensions except the concatenated ones
    out_dims_set = set(args[0][0].dims) - {axes[0]}
    for axis, arg in zip(axes[1:], args[0][1:]):
        arg_dims_set = set(arg.dims) - {axis}
        if arg_dims_set != out_dims_set:
            raise ValueError(f"Arrays must have all the same dimensions except those concatentated over. The first input array {args[0][0]} has non-concatenated dimensions {out_dims_set}, while the input array {arg} has non-concatenated dimensions {arg_dims_set}.")
    # Align the arrays. Use the shape of the first array as the template.
    ambiguous_out_dims = {dim for arg in args[0] for dim in arg.ambiguous_dims if dim not in axes}
    aligned_args = [args[0][0]]
    for axis, arg in zip(axes[1:], args[0][1:]):
        for dim0, dim in zip(args[0][0].dims, arg.dims):
            if axes[0] != dim0 and dim != axis and dim0 != dim:
                ambiguous_out_dims.add(dim)
        aligned_args.append(arg[tuple(dim if dim != axes[0] else axis for dim in args[0][0].dims)])
    out_dims = [dim if dim != axes[0] else tuple(axes) for dim in args[0][0].dims]
    axis_num = args[0][0].dims.index(axes[0])
    return einexpr.einarray(func(tuple(arg.a for arg in aligned_args), *args[1:], axis=axis_num, **kwargs), dims=out_dims, ambiguous_dims=ambiguous_out_dims)


typical_function_dispatches = dict(
    # Module functions
    **{name: single_arg_elementwise_dispatch for name in "sign sqrt inv where clip argsort ones_like zeros_like full_like softmax log_softmax cumsum abs absolute sin cos tan exp log exp2 log2 log10 sqrt pow reciprocal floor ceil round floor_divide".split()},
    **{name: multi_dim_reduction_dispatch for name in "sum product mean std max min maximum minimum all any".split()},
    **{name: single_dim_reduction_dispatch for name in "argmax argmin".split()},
    **{name: concatenation_dispatch for name in "concat concatenate".split()},
    # Array functions
    **{f'__{name}__': single_arg_elementwise_dispatch for name in "neg pos invert".split()},
    **{f'__{name}__': multi_arg_elementwise_dispatch for name in "add radd sub rsub mul rmult div rdiv truediv rtruediv mod rmod pow rpow eq".split()},
)


FunctionRegistration = namedtuple('FunctionRegistration', ['name', 'base_function', 'dispatch', 'backend'])

def register_function(name: str, base_function: Callable, dispatch: Callable, backend: str):
    function_registry.append(FunctionRegistration(name, base_function, dispatch, backend))
    if callable(base_function):
        function_registry.append(FunctionRegistration(name, base_function.__call__, dispatch, backend))  # Register __call__ too for 'functions' like numpy ufuncs that are actually objects.


def search_function_registry(**kwargs):
    """
    Yields all function registrations that match the given kwargs.
    """
    global function_registry
    # Ensure the search fields are valid
    if len(kwargs) == 0:
        raise ValueError("No search fields provided")
    for key in kwargs:
        if key not in FunctionRegistration._fields:
            raise ValueError(f"Invalid search field {key}")
    for registration in function_registry:
        if all(hasattr(registration, k) and getattr(registration, k) == v for k, v in kwargs.items()):
            yield registration


def register_typical_functions(np_like, backend):
    """
    Registers all functions in np_like with a typical name (like 'sum', 'concat', etc.).
    """
    for name, function in np_like.__dict__.items():
        if isinstance(function, Callable):
            if name in typical_function_dispatches:
                if next(search_function_registry(base_function=function), None) is None:
                    register_function(name, function, typical_function_dispatches[name], backend)


# numpy
import numpy as np

backend_array_types['numpy'] = [np.ndarray]

# Register all universal functions as defaults by iterating over functions
for function_name, function in np.__dict__.items():
    if isinstance(function, np.ufunc) and function.signature is None:
        register_function(function_name, None, multi_arg_elementwise_dispatch, 'default')

register_typical_functions(np, 'numpy')
register_typical_functions(np.ndarray, 'numpy')


# PyTorch
if importlib.util.find_spec('torch'):
    import torch

    pytorch_tensor_types = [
        torch.BFloat16Tensor,
        torch.BoolTensor,
        torch.ByteTensor,
        torch.CharTensor,
        torch.DoubleTensor,
        torch.FloatTensor,
        torch.HalfTensor,
        torch.IntTensor,
        torch.LongTensor,
        torch.ShortTensor,
        torch.Tensor,
        torch.TensorType]

    backend_array_types['torch'] = pytorch_tensor_types
    register_typical_functions(torch, 'torch')
    for array_type in pytorch_tensor_types:
        register_typical_functions(array_type, 'torch')

# JAX
if importlib.util.find_spec('jax'):
    import jax
    import jax.numpy as jnp

    jax_array_types = [
        jax.numpy.ndarray,
        jax.interpreters.xla.DeviceArray, 
        jax.interpreters.pxla.ShardedDeviceArray, 
        jax.interpreters.partial_eval.DynamicJaxprTracer,
        jax.interpreters.batching.BatchTracer,
        jax.interpreters.batching.Tracer,
    ]

    JAXArrayLike = jax_array_types
    backend_array_types['jax'] = JAXArrayLike

    register_typical_functions(jnp, 'jax')
    for array_type in jax_array_types:
        register_typical_functions(array_type, 'jax')


# Pseudo-backend for dimension inference
class PseudoRawArray:
    def __array__(self):
        raise NotImplementedError("Attempted to use a PseudoRawArray as an array")
    
    def __array_function__(self, func, types, args, kwargs):
        return self
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        raise self
    
    def __torch_function__(self, func, types, args, kwargs):
        raise self
    
    def __repr__(self):
        return "PseudoRawArray()"


register_typical_functions(PseudoRawArray, 'pseudo')

# End of backend-specific sections
RawArrayLike = Union[tuple(array_type for _backend_array_types in backend_array_types.values() for array_type in _backend_array_types) + (PseudoRawArray,)]
def detect_backend(array: RawArrayLike) -> str:
    for backend, array_type in backend_array_types.items():
        if isinstance(array, array_type):
            return backend
    raise ValueError(f"Could not detect backend for array {array}")



