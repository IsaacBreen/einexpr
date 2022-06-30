from collections import namedtuple
from pyclbr import Function
from typing import Union, Callable
import numpy as np
import string

backend_array_types = dict()
backend_dim_kwargs_to_resolve = ['dim', 'axis']

function_registry = []

typical_function_classes = dict(
    elementwise="sign sqrt inv where clip argsort ones_like zeros_like full_like softmax log_softmax cumsum".split(),
    multi_dim_reduction="sum product mean std max min maximum minimum all any ".split(),
    single_dim_reduction="argmax argmin".split(),
    concatenation="concat concatenate".split(),
)


def elementwise_signature(args, kwargs):
    return f"{','.join(['()'] * len(args))}->()"


def multi_dim_reduction_signature(args, kwargs):
    assert len(args) == 1
    args = args[0]
    assert len(args) == 1
    in_dims = list(string.ascii_lowercase[:len(args[0].dims)])
    if 'axis' in kwargs:
        axis = kwargs['axis']
    elif 'dims' in kwargs:
        axis = kwargs['dims']
    else:
        axis = None
    if axis is None:
        axis = list(range(in_dims))
    else:
        if isinstance(axis, int):
            axis = [axis]
        axis = [i if i >= 0 else i + len(in_dims) for i in axis]
    out_dims = [dims for i, dims in enumerate(in_dims) if i not in axis]
    return f"({','.join(out_dims)})->()"


def single_dim_reduction_signature(args, kwargs):
    if 'axis' in kwargs:
        axis = kwargs['axis']
    elif 'dims' in kwargs:
        axis = kwargs['dims']
    else:
        axis = None
    assert axis is None or isinstance(axis, int)
    return multi_dim_reduction_signature(args, kwargs)


def concatenation_signature(args, kwargs):
    assert len(args) == 1
    args = args[0]
    axis = kwargs.get('axis', None) or kwargs.get('dim', None)
    if 'axis' in kwargs:
        axis = kwargs['axis']
    elif 'dims' in kwargs:
        axis = kwargs['dims']
    else:
        axis = None
    if axis is None:
        axis = 0
    non_concatenated_dims = args[0].dims[:axis] + args[0].dims[axis+1:]
    for arg in args[1:]:
        assert non_concatenated_dims == arg.dims[:axis] + arg.dims[axis+1:]
    aliases = iter(string.ascii_lowercase)
    left_non_concatenated_dims = [next(aliases) for _ in range(axis)]
    right_non_concatenated_dims = [next(aliases) for _ in range(axis+1, len(args[0].dims))]
    inp_signatures = []
    for arg in args:
        inp_signatures.append('(' + ','.join((*left_non_concatenated_dims, next(aliases), *right_non_concatenated_dims)) + ')')
    out_signature = '(' + ','.join((*left_non_concatenated_dims, next(aliases), *right_non_concatenated_dims)) + ')'
    return ','.join(inp_signatures) + '->' + out_signature


FunctionRegistration = namedtuple('FunctionRegistration', ['name', 'function', 'signature', 'backend'])


def register_function(name: str, function: Callable, signature: Union[str, Callable], backend: str):
    function_registry.append(FunctionRegistration(name, function, signature, backend))


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


def register_functions_to_default(np_like, backend):
    """
    For each function in np_like, search the function registry for a registration with the same name but with a backend of the value 'default'. If such a registration is found, register it with the new backend.
    """
    for registration in search_function_registry(backend='default'):
        if hasattr(np_like, registration.name):
            # Get the first registration that matches
            if registration is not None:
                register_function(registration.name, getattr(np_like, registration.name), registration.signature, backend)


# Register the typical functions as defaults
for function_name in typical_function_classes['elementwise']:
    register_function(function_name, None, elementwise_signature, 'default')

for function_name in typical_function_classes['multi_dim_reduction']:
    register_function(function_name, None, multi_dim_reduction_signature, 'default')
    
for function_name in typical_function_classes['single_dim_reduction']:
    register_function(function_name, None, single_dim_reduction_signature, 'default')
    
for function_name in typical_function_classes['concatenation']:
    register_function(function_name, None, concatenation_signature, 'default')


# numpy
import numpy as np

backend_array_types['numpy'] = np.ndarray

# Register all universal functions as defaults by iterating over functions
for function_name, function in np.__dict__.items():
    if isinstance(function, np.ufunc):
        register_function(function_name, None, function.signature, 'default')

register_functions_to_default(np, 'numpy')

# PyTorch
import torch

backend_array_types['torch'] = torch.Tensor
register_functions_to_default(torch, 'torch')

# JAX
import jax
import jax.numpy as jnp

JAXArrayLike = Union[
    jax.interpreters.xla.DeviceArray, 
    jax.interpreters.pxla.ShardedDeviceArray, 
    jax.interpreters.partial_eval.DynamicJaxprTracer,
    jax.interpreters.batching.BatchTracer,
]

backend_array_types['jax'] = JAXArrayLike
register_functions_to_default(jnp, 'jax')

# End of backend-specific sections

RawArrayLike = Union[(*backend_array_types.values(),)]

def detect_backend(array: RawArrayLike) -> str:
    for backend, array_type in backend_array_types.items():
        if isinstance(array, array_type):
            return backend
    raise ValueError(f"Could not detect backend for array {array}")