from typing import Union

backend_array_types = dict()

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