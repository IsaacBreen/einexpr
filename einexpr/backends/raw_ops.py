from itertools import zip_longest
from typing import Any, List, Optional, Sequence, Tuple, Union, Container, Iterator

import numpy as np

import einexpr
from .backends import RawArrayLike


def apply_transexpand(array: RawArrayLike, transexpansion: List[int]) -> 'einexpr.einarray':
    """
    Returns a new array with the given transpose.
    """
    if not transexpansion:
        return array
    else:
        transposition = [i for i in transexpansion if i is not None]
        expansion = tuple(None if i is None else slice(None) for i in transexpansion)
        return array.__array_namespace__().permute_dims(array, transposition)[expansion]


def align_to_dims(a: 'einexpr.einarray', dims: List['einexpr.Dimension'], expand: bool = False) -> RawArrayLike:
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
        return apply_transexpand(a.__array__(), einexpr.dimension_utils.calculate_transexpand([dim for dim in a.dims if dim in dims], dims))
    
    
def align_arrays(*arrays: 'einexpr.einarray', return_output_dims: bool = False) -> Union[List[RawArrayLike], Tuple[List[RawArrayLike], List['einexpr.Dimension']]]:
    """
    Aligns the given arrays to common dimensions.
    """
    if not arrays:
        return [], [] if return_output_dims else []
    else:
        # Calculate the output dimensions
        out_dims = einexpr.dimension_utils.get_final_aligned_dims(*(array.dims for array in arrays))
        # Calculate and apply the transpositions and expansions necessesary to align the arrays to the output dimensions
        raw_aligned_arrays = []
        for array in arrays:
            if isinstance(array.a, (int, float)):
                raw_aligned_arrays.append(array.a)
            else:
                raw_aligned_arrays.append(apply_transexpand(array.a, einexpr.dimension_utils.calculate_transexpand(array.dims, out_dims)))
        if return_output_dims:
            return raw_aligned_arrays, out_dims
        else:
            return raw_aligned_arrays


def reduce_sum(array: 'einexpr.einarray', dims: Union[Container['einexpr.Dimension'], Iterator['einexpr.Dimension']]) -> 'einexpr.einarray':
    """
    Collapse the given array along the given dimensions.
    """
    if not set(array.dims) & set(dims):
        return array
    if set(dims) - set(array.dims):
        raise ValueError(f"The dimensions {dims} must be a subset of the dimensions {a.dims} of the input array.")
    return einexpr.einarray(
        array.a.__array_namespace__().sum(array.a, axis=tuple(i for i, dim in enumerate(array.dims) if dim in dims)),
        dims=[dim for dim in array.dims if dim not in dims], ambiguous_dims={dim for dim in array.dims if dim not in dims}
    )