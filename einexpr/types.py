from typing import *
import numpy as np
import numpy.typing as npt

RawArray = npt.ArrayLike


class ConcreteArrayLike:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError


class Dimension:
    def __init__(self, *args, **kwargs):
    
        raise NotImplementedError

class LazyArrayLike:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError


RawArrayLike = Union[int, float, np.ndarray]
    
ArrayLike = Union[ConcreteArrayLike, LazyArrayLike]