from itertools import zip_longest
from typing import List, Optional, Tuple, Sequence, Any, Union

import numpy as np

from einexpr.types import Dimension
from .utils import powerset
from .parse_numpy_ufunc_signature import UfuncSignature, UfuncSignatureDimensions, parse_ufunc_signature
from .types import ConcreteArrayLike, RawArray
from .dim_calcs import calculate_transexpand, dims_after_alignment, calculate_signature_transexpands


def apply_transexpand(a: RawArray, transpose: List[int]) -> ConcreteArrayLike:
    """
    Returns a new array with the given transpose.
    """
    if not transpose:
        return a
    else:
        return np.expand_dims(np.transpose(a, [i for i in transpose if i is not None]), [n for n, i in enumerate(transpose) if i is None])


def align_to_dims(a: ConcreteArrayLike, dims: Sequence[Dimension], expand: bool = False) -> RawArray:
    """
    Aligns the given arrays to the given dimensions and expands along dimensions that each input array does not have.
    """
    if tuple(a.dims) == tuple(dims):
        return a.__array__()
    else:
        if not expand and set(dims) - set(a.dims):
            raise ValueError(f"The dimensions {dims} requested must be a permutation of the dimensions {a.dims} of the input array. To expand new dimensions, pass expand=True.")
        # if not ignore_extra_dims and set(a.dims) - set(dims):
        #     raise ValueError(f"The dimensions {a.dims} of the input array must be a subset of the requested dimensions {dims}. To ignore extra dimensions, pass ignore_extra_dims=True.")
        return apply_transexpand(a.__array__(), calculate_transexpand([dim for dim in a.dims if dim in dims], dims))
    
    
def align_arrays(*arrays: ConcreteArrayLike, signature: Optional[UfuncSignature] = None, return_dims: bool = False) -> Union[List[RawArray], Tuple[List[RawArray], List[Dimension]]]:
    """
    Aligns the given arrays to common dimensions.
    """
    if not arrays:
        return [], [] if return_dims else []
    else:
        dims = dims_after_alignment(*[array.dims for array in arrays])
        raw_arrays = [align_to_dims(array, dims, expand=True) for array in arrays]
        assert all(array.ndim == len(dims) for array in raw_arrays)
        return raw_arrays, dims if return_dims else raw_arrays