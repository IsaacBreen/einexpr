from types import GeneratorType
from typing import Tuple
import einexpr

class PseudoRawArray:
    shape: Tuple[int, ...]
    dtype: type
    
    def __init__(self, shape, dtype=None):
        self.shape = shape
        self.dtype = dtype
    
    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        else:
            return self
    

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