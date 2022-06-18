from itertools import zip_longest
from typing import List, Tuple, Sequence, Any, Union

import numpy as np

from einexpr.types import Dimension
from .utils import powerset
from .parse_numpy_ufunc_signature import UfuncSignature, UfuncSignatureDimensions, parse_ufunc_signature
from .types import ConcreteArrayLike, RawArray
from .dim_calcs import calculate_transpose, dims_after_alignment


def apply_transpose(a: RawArray, transpose: List[int]) -> ConcreteArrayLike:
    """
    Returns a new array with the given transpose.
    """
    if not transpose:
        return a
    else:
        return np.expand_dims(np.transpose(a, [i for i in transpose if i is not None]), [j for i,j in enumerate(transpose) if i is None])


def align_to_dims(a: ConcreteArrayLike, dims: Sequence[Dimension]) -> RawArray:
    """
    Aligns the given arrays to the given dimensions.
    """
    if len(a.dims) == len(dims):
        return a.__array__()
    else:
        return apply_transpose(a.__array__(), calculate_transpose(a.dims, dims))
    
    
def align_arrays(*arrays: ConcreteArrayLike, return_dims=False) -> Union[List[RawArray], Tuple[List[RawArray], List[Dimension]]]:
    """
    Aligns the given arrays to the same dimensions.
    """
    if not arrays:
        return [], [] if return_dims else []
    else:
        dims = dims_after_alignment(*[array.dims for array in arrays])
        raw_arrays = [align_to_dims(array, dims) for array in arrays]
        return raw_arrays, dims if return_dims else raw_arrays