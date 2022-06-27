from typing import *
from .backends import RawArrayLike

class ConcreteArrayLike:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError


class Dimension:
    def __init__(self, *args, **kwargs):
    
        raise NotImplementedError

class LazyArrayLike:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

DimensionlessLike = Union[int, float]
    
ArrayLike = Union[ConcreteArrayLike, LazyArrayLike]
ConcreteLike = Union[DimensionlessLike, ConcreteArrayLike]