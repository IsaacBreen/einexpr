from typing import *
import numpy.typing as npt

RawArray = npt.ArrayLike
ConcreteArrayLike = TypeVar('ConcreteArrayLike', bound=RawArray)
Dimension = TypeVar('Dimension', bound=str)
LazyArrayLike = TypeVar('LazyArrayLike')
ArrayLike = Union[ConcreteArrayLike, LazyArrayLike]