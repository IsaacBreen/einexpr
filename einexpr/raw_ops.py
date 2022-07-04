from itertools import zip_longest
from typing import List, Optional, Tuple, Sequence, Any, Union

import numpy as np

from einexpr.base_typing import Dimension
from .utils import powerset
from .parse_numpy_ufunc_signature import UfuncSignature, UfuncSignatureDimensions, parse_ufunc_signature
from .typing import ConcreteArrayLike, RawArrayLike
from .dim_calcs import *


def apply_transexpand(a: RawArrayLike, transexpansion: List[int]) -> ConcreteArrayLike:
    """
    Returns a new array with the given transpose.
    """
    if not transexpansion:
        return a
    else:
        transposition = [i for i in transexpansion if i is not None]
        expansion = tuple(None if i is None else slice(None) for i in transexpansion)
        return a.transpose(transposition)[expansion]


def align_to_dims(a: ConcreteArrayLike, dims: List[Dimension], expand: bool = False) -> RawArrayLike:
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
    
    
def align_arrays(*arrays: ConcreteArrayLike, return_output_dims: bool = False) -> Union[List[RawArrayLike], Tuple[List[RawArrayLike], List[Dimension]]]:
    """
    Aligns the given arrays to common dimensions.
    """
    if not arrays:
        return [], [] if return_output_dims else []
    else:
        # Calculate the output dimensions
        out_dims = get_final_aligned_dims(*(array.dims for array in arrays))
        # Calculate and apply the transpositions and expansions necessesary to align the arrays to the output dimensions
        raw_aligned_arrays = [apply_transexpand(array.__array__(), calculate_transexpand(array.dims, out_dims)) for array in arrays]
        if return_output_dims:
            return raw_aligned_arrays, out_dims
        else:
            return raw_aligned_arrays