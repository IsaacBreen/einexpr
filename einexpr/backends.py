from typing import Union, Callable

backend_array_types = dict()
backend_dim_kwargs_to_resolve = ['dim', 'axis']

function_types = {
}

typical_function_classes = dict(
    elementwise="sign sqrt inv where clip argsort ones_like zeros_like full_like softmax log_softmax cumsum".split(),
    multi_dim_reduction="sum product mean std max min maximum minimum all any ".split(),
    single_dim_reduction="argmax argmin".split(),
    concatenation="concat concatenation".split(),
)

def register_function(function_type: str, function: Callable):
    function_types.setdefault(function_type, set()).add(function)

def register_functions(**kwargs):
    for function_type, function in kwargs.items():
        register_function(function, function_type)

def register_typical_functions(np_like):
    for function_class, functions in typical_function_classes.items():
        for function in functions:
            if hasattr(np_like, function):
                register_function(function_class, getattr(np_like, function))

# numpy
import numpy as np

backend_array_types['numpy'] = np.ndarray



# PyTorch
import torch

backend_array_types['torch'] = torch.Tensor

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

# End of backend-specific sections

RawArrayLike = Union[(*backend_array_types.values(),)]

def detect_backend(array: RawArrayLike) -> str:
    for backend, array_type in backend_array_types.items():
        if isinstance(array, array_type):
            return backend
    raise ValueError(f"Could not detect backend for array {array}")