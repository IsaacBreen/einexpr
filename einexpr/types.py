from typing import *
from numpy.typing import ArrayLike

RawArray = ArrayLike
ConcreteArrayLike = TypeVar('ConcreteArrayLike', bound=RawArray)
Dimension = TypeVar('Dimension', bound=str)
LazyArrayLike = TypeVar('LazyArrayLike')
