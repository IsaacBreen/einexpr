import importlib
from collections import namedtuple
from typing import Any, Callable, Union


from einexpr.dimension_utils import (
    SingleArgumentElementwise,
    MultiArgumentElementwise,
    MultiDimensionReduction,
    SingleDimensionReduction,
    Concatenation,
)

backend_array_types = dict()
backend_dim_kwargs_to_resolve = ['dim', 'axis']

function_registry = []


def single_arg_elementwise_dispatch(func, args, kwargs):
    return temp_dispatch_integrator(SingleArgumentElementwise, func, args, kwargs)


def multi_arg_elementwise_dispatch(func, args, kwargs):
    return temp_dispatch_integrator(MultiArgumentElementwise, func, args, kwargs)

def multi_dim_reduction_dispatch(func, args, kwargs):
    return temp_dispatch_integrator(MultiDimensionReduction, func, args, kwargs)

def single_dim_reduction_dispatch(func, args, kwargs):
    return temp_dispatch_integrator(SingleDimensionReduction, func, args, kwargs)

def concatenation_dispatch(func, args, kwargs):
    return temp_dispatch_integrator(Concatenation, func, args, kwargs)


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
    def __getattr__(self, name):
        return PseudoRawArray()
    
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


def conforms_to_array_api(array: Any):
    """
    Returns True if the given array conforms to the array API.
    """
    return hasattr(array, '__array_namespace__')