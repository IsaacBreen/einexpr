import einexpr
from typing import *


class ConcreteArrayLike:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError


Dimension = str


class LazyArrayLike:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

DimensionlessLike = Union[int, float]
    
EinarrayLike = Union[ConcreteArrayLike, LazyArrayLike]
ConcreteLike = Union[DimensionlessLike, ConcreteArrayLike]