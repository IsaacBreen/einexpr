from types import GeneratorType
from typing import Tuple
import einexpr


class PseudoArrayNamespace:
    def __getattr__(self, name):
        return lambda *args, **kwargs: PseudoRawArray()


class PseudoRawArray:
    _shape: Tuple[int, ...]
    _dtype: type
    
    def __init__(self, shape=None, dtype=None):
        self._shape = shape
        self._dtype = dtype
    
    @property
    def shape(self):
        if self._shape is None:
            raise ValueError("shape is not set")
        return self._shape

    @property
    def dtype(self):
        if self._dtype is None:
            raise ValueError("dtype is not set")
        return self._dtype
    
    @property
    def ndim(self):
        return len(self.shape)

    def __array_namespace__(self):
        return PseudoArrayNamespace()

def apply_tracer(obj):
    if isinstance(obj, PseudoRawArray):
        return obj
    elif isinstance(obj, list):
        return [apply_tracer(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(apply_tracer(item) for item in obj)
    elif isinstance(obj, dict):
        return {apply_tracer(key): apply_tracer(value) for key, value in obj.items()}
    elif isinstance(obj, einexpr.einarray):
        return obj.tracer()
    elif isinstance(obj, GeneratorType):
        return (apply_tracer(item) for item in obj)
    else:
        return obj