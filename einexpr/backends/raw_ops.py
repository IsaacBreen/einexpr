from itertools import zip_longest
from typing import Any, Iterable, List, Optional, Sequence, Tuple, TypeVar, Union, Container, Iterator

import numpy as np

import einexpr

RawArray = TypeVar('RawArray')


def apply_transexpand(array: RawArray, transexpansion: List[int]) -> RawArray:
    """
    Returns a new array with the given transpose.
    """
    if not transexpansion:
        return array
    else:
        transposition = [i for i in transexpansion if i is not None]
        expansion = tuple(None if i is None else slice(None) for i in transexpansion)
        return array.__array_namespace__().permute_dims(array, transposition)[expansion]


def reshape(array: RawArray, shape: Tuple[int, ...]) -> RawArray:
    """
    Reshape the given array to the given shape.
    """
    return array.__array_namespace__().reshape(array, shape)


def align_to_dimspec(
    raw_array: RawArray,
    current_dimspec: einexpr.array_api.dimension.DimensionSpecification,
    target_dimspec: einexpr.array_api.dimension.DimensionSpecification
) -> RawArray:
    """
    Aligns the given arrays to the given dimensions, collapses along dimensions that appear in the input array but not the ``dims`` argument, and expands along dimensions that the input array does not have.
    """
    # Note: there are two kind of expansions referred to below: expansions as in reshaping ``(i j)`` into ``i j``, and
    # expansions as in using expand_dims to creaate a new dimension ``k`` of size 1 such that ``i j`` becomes ``i j k``.
    #
    # Suppose ``x`` is an einarray that we want to reshape from ``i (j k) l`` to ``(i k) j m``, where ``m`` is a new dimension.
    # The steps are as follows:
    # 1. Fully expand the input array: ``x['i (j k) l')]`` -> ``x['i j k l']``
    # 2. Fully expand the output dimensions: ``(i k) j m`` -> ``i j k m``
    # 3. Collapse the expanded array along dimension ``l``, which is not in the output dimensions: ``x['i j k l']`` -> ``x['i j k']``
    # 4. Permute the dimensions of the collapsed array into the shape given by step 2: ``x['i j k']`` -> ``x['i k j']``
    # 5. Create a new dimension ``m`` of size 1: ``x['i k j']`` -> ``x['i k j m']``
    # 6. Reshape to combine dimensions as required: ``x['i k j m']`` -> ``x['(j k) i m']``
    #
    # 1. Fully expand the raw array
    expanded_current_dimspec = einexpr.dimension_utils.expand_dims(current_dimspec)
    expanded_shape = einexpr.dimension_utils.dimspec_to_shape(expanded_current_dimspec)
    raw_array = reshape(raw_array, expanded_shape)
    assert raw_array.shape == expanded_shape
    # 2. Fully expand the output dimensions
    expanded_new_dims = einexpr.dimension_utils.expand_dims(target_dimspec)
    # 3. Collapse along dimensions that are not in the output dimensions
    dims_to_collapse = set(expanded_current_dimspec) - set(expanded_new_dims)
    if dims_to_collapse:
        raw_array = einexpr.sum(einexpr.einarray(raw_array, dims=expanded_current_dimspec), axis=dims_to_collapse).a
    collapsed_expanded_current_dimspec = tuple(dim for dim in expanded_current_dimspec if dim not in dims_to_collapse)
    # 4 & 5. Permute the dimensions of the raw array and create a new dimension in one go
    transexpand = einexpr.dimension_utils.calculate_transexpand(collapsed_expanded_current_dimspec, expanded_new_dims)
    raw_array = apply_transexpand(raw_array, transexpand)
    # 6. Combine as required
    final_shape = einexpr.dimension_utils.dimspec_to_shape(target_dimspec)
    raw_array = reshape(raw_array, final_shape)
    return raw_array


def align_arrays(*arrays: 'einexpr.einarray', return_output_dims: bool = False) -> Union[List[einexpr.types.NonEinArray], Tuple[List[einexpr.types.NonEinArray], List['einexpr.array_api.dimension.AtomicDimension']]]:
    """
    Aligns the given arrays to common dimensions.
    """
    if not arrays:
        return [], [] if return_output_dims else []
    else:
        # Calculate the output dimensions
        out_dims = einexpr.dimension_utils.get_final_aligned_dims(*(einexpr.dimension_utils.get_dims(array) for array in arrays))
        # Calculate and apply the transpositions and expansions necessesary to align the arrays to the output dimensions
        raw_aligned_arrays = []
        for array in arrays:
            if isinstance(einexpr.dimension_utils.get_raw(array), (int, float)):
                raw_aligned_arrays.append(einexpr.dimension_utils.get_raw(array))
            else:
                transexpand = einexpr.dimension_utils.calculate_transexpand(einexpr.dimension_utils.get_dims(array), out_dims)
                raw_aligned_arrays.append(apply_transexpand(einexpr.dimension_utils.get_raw(array), transexpand))
        if return_output_dims:
            return raw_aligned_arrays, out_dims
        else:
            return raw_aligned_arrays