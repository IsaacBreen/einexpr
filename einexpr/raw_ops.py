from itertools import zip_longest
from typing import List, Optional, Tuple, Sequence, Any, Union

import numpy as np

from einexpr.types import Dimension
from .utils import powerset
from .parse_numpy_ufunc_signature import UfuncSignature, UfuncSignatureDimensions, parse_ufunc_signature
from .types import ConcreteArrayLike, RawArray
from .dim_calcs import calculate_output_dims_from_signature, calculate_transexpand, dims_after_alignment, calculate_signature_transexpands


def apply_transexpand(a: RawArray, transpose: List[int]) -> ConcreteArrayLike:
    """
    Returns a new array with the given transpose.
    """
    if not transpose:
        return a
    else:
        transpose_arg = [i for i in transpose if i is not None]
        expand_arg = [n for n, i in enumerate(transpose) if i is None]
        return np.expand_dims(np.moveaxis(a, transpose_arg, range(len(transpose_arg))), expand_arg)


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
    
    
def align_arrays(*arrays: ConcreteArrayLike, signature: Optional[UfuncSignature] = None, return_output_dims: bool = False) -> Union[List[RawArray], Tuple[List[RawArray], List[Dimension]]]:
    """
    Aligns the given arrays to common dimensions.
    """
    if not arrays:
        return [], [] if return_output_dims else []
    else:
        # Calculate the transpositions and expansions necessesary to align the arrays
        transexpands = calculate_signature_transexpands(signature, *(a.dims for a in arrays))
        # Apply these
        raw_aligned_arrays = [apply_transexpand(a.__array__(), transexpand) for a, transexpand in zip(arrays, transexpands)]
        if return_output_dims:
            return raw_aligned_arrays, calculate_output_dims_from_signature(signature, *(a.dims for a in arrays))
        else:
            return raw_aligned_arrays