import collections
import functools
import operator
from os import PRIO_PGRP
import re
from itertools import chain, zip_longest
from types import EllipsisType
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, TypeVar, Union

import numpy as np
from pydantic import PositiveInt
from einexpr.array_api.dimension import NamedDimension, PositionalDimension

from einexpr.utils.utils import pedantic_dict_merge
from .exceptions import AmbiguousDimensionException
import einexpr
from lark import Lark, Transformer, v_args

from dataclasses import replace


D = TypeVar("D", bound=einexpr.array_api.dimension.DimensionObject)


def get_unambiguous_broadcast_dims(
    *dims: List[einexpr.array_api.dimension.Dimension],
) -> Set[einexpr.array_api.dimension.Dimension]:
    scs = iter(einexpr.utils.get_all_scs_with_unique_elems(*dims))
    first_broadcasted_dims = np.array(list(next(scs)))
    unambiguous_positions = np.ones((len(first_broadcasted_dims),), dtype=bool)
    for broadcasted_dims in scs:
        unambiguous_positions &= first_broadcasted_dims == np.array(
            list(broadcasted_dims)
        )
    return set(first_broadcasted_dims[unambiguous_positions].tolist())


def get_ambiguous_broadcast_dims(
    *dims: List[einexpr.array_api.dimension.Dimension],
) -> Set[einexpr.array_api.dimension.Dimension]:
    scs = iter(einexpr.utils.get_all_scs_with_unique_elems(*dims))
    first_broadcasted_dims = np.array(list(next(scs)))
    ambiguous_positions = np.zeros((len(first_broadcasted_dims),), dtype=bool)
    for broadcasted_dims in scs:
        ambiguous_positions |= first_broadcasted_dims != np.array(
            list(broadcasted_dims)
        )
    return set(first_broadcasted_dims[ambiguous_positions].tolist())


def get_unique_broadcast(
    *dims: List[einexpr.array_api.dimension.Dimension],
) -> Set[einexpr.array_api.dimension.Dimension]:
    scs = einexpr.utils.get_all_scs_with_unique_elems(*dims)
    if len(scs) == 0:
        raise ValueError("No valid broadcast found.")
    if len(scs) > 1:
        raise AmbiguousDimensionException("Multiple valid broadcasts found.")
    return scs.pop()


def get_any_broadcast(
    *dims: List[einexpr.array_api.dimension.Dimension],
) -> Set[einexpr.array_api.dimension.Dimension]:
    """
    Use this when you need a broadcast but don't care about its uniqueness. The broadcast chosen is deterministic and depends only on the order dimensions within each argument and the order in which the arguments are passed.
    """
    scs = einexpr.utils.get_all_scs_with_unique_elems(*dims)
    if len(scs) == 0:
        raise ValueError("No valid broadcast found.")
    return sorted(scs).pop()


def get_final_aligned_dims(
    *ein_dims: List[einexpr.array_api.dimension.Dimension],
) -> List[einexpr.array_api.dimension.Dimension]:
    """
    Aligns and broadcasts the given dimensions.
    """
    # for dim in iter_dims(ein_dims):
    #     if isinstance(dim, einexpr.array_api.dimension.PositionalDimension):
    #         raise TypeError("Convert positional dimensions to named dimensions first.")
    scs = einexpr.utils.get_all_scs_with_nonunique_elems_removed(*ein_dims)
    if len(scs) == 0:
        raise ValueError("No valid alignment found.")
    return sorted(scs, key=lambda x: map_over_dims(lambda dim: dim.id, x)).pop()


def calculate_ambiguous_final_aligned_dims(
    *dims: List[einexpr.array_api.dimension.Dimension],
) -> Set[einexpr.array_api.dimension.Dimension]:
    aligned_dims = get_each_aligned_dims(*dims)
    scs = list(einexpr.utils.get_all_scs_with_unique_elems(*aligned_dims))
    if len(scs) == 0:
        raise ValueError(f"No valid alignments for the given dimensions {dims}.")
    first_broadcasted_dims = np.array(scs[0])
    ambiguous_positions = np.zeros((len(first_broadcasted_dims),), dtype=bool)
    for broadcasted_dims in scs[1:]:
        ambiguous_positions |= first_broadcasted_dims != np.array(
            list(broadcasted_dims)
        )
    return set(first_broadcasted_dims[ambiguous_positions].tolist())


def get_each_aligned_dims(
    *ein_dims: List[einexpr.array_api.dimension.Dimension],
) -> List[List[einexpr.array_api.dimension.Dimension]]:
    """
    Returns a list of lists of dimensions that are aligned with eachother.
    """
    aligned_final = get_final_aligned_dims(*ein_dims)
    return [[dim for dim in aligned_final if dim in dims] for dims in ein_dims]


def calculate_transexpand(
    dims_from: List[einexpr.array_api.dimension.Dimension],
    dims_to: List[einexpr.array_api.dimension.Dimension],
) -> List[einexpr.array_api.dimension.Dimension]:
    """
    Returns lists of the transpositions and expansions required to align to given dimensions. Broadcast dimensions - those
    that will need to be expanded - are represented by None.
    """
    return [dims_from.index(dim) if dim in dims_from else None for dim in dims_to]


@v_args(inline=True)
class TreeToReshapeDimension(Transformer):
    def start(self, root_dims) -> Any:
        return root_dims

    def root_dims(self, dim: D) -> D:
        return dim

    def replacable_dim(self, value) -> Union[einexpr.array_api.dimension.Dimension, einexpr.array_api.dimension.DimensionTuple]:
        return value
        
    def dim_replacement(self, original, replacement) -> einexpr.array_api.dimension.DimensionReplacement:
        return einexpr.array_api.dimension.DimensionReplacement(original, replacement)

    def irreplacable_dim(self, value) -> Union[einexpr.array_api.dimension.Dimension, einexpr.array_api.dimension.DimensionTuple]:
        if isinstance(value, str):
            return value
        elif isinstance(value, tuple):
            return value
        else:
            raise ValueError(f"Unexpected value {value}.")

    def tuple(self, expand_sentinel: einexpr.utils.ExpandSentinel) -> einexpr.array_api.dimension.DimensionTuple:
        return einexpr.array_api.dimension.DimensionTuple(expand_sentinel.content)
    
    def bare_tuple(self, *dims) -> einexpr.utils.ExpandSentinel:
        return einexpr.utils.ExpandSentinel(tuple(dim for dim in dims if dim is not None))
    
    def ellipsis_bare_tuple(self, *dims) -> einexpr.utils.ExpandSentinel:
        return einexpr.utils.ExpandSentinel(tuple(dim for dim in dims if dim is not None))
    
    def sequence(self, *xs) -> Tuple:
        return einexpr.array_api.dimension.DimensionTuple(x for x in xs if x is not None)

    def sep(self) -> None:
        return None
    
    def named_dim(self, value: str) -> einexpr.array_api.dimension.NamedDimension:
        return einexpr.array_api.dimension.NamedDimension(value)
    
    def positional_dim(self, value: str) -> einexpr.array_api.dimension.PositionalDimension:
        return einexpr.array_api.dimension.PositionalDimensionPlaceholder()

    def EMPTY(self, tree) -> str:
        return tree.value
    
    def NAME(self, tree) -> str:
        return tree.value
    
    def ELLIPSIS(self, tree) -> EllipsisType:
        return Ellipsis


dims_reshape_grammar = Lark(
    r"""
        %import common.WS

        %ignore WS

        start: root_dims
        ?root_dims: replacable_dim | ellipsis_bare_tuple{replacable_dim} | bare_tuple{replacable_dim}
        ?replacable_dim: dim | "(" replacable_dim ")" | dim_replacement | tuple{replacable_dim}
        dim_replacement: irreplacable_dim "->" irreplacable_dim
        ?irreplacable_dim: dim | "(" irreplacable_dim ")" | tuple{irreplacable_dim}
        tuple{x}: "(" bare_tuple{x} ")"
        bare_tuple{x}: x sep | x [sep] (x [sep])+
        ellipsis_bare_tuple{x}: (bare_tuple{x} | replacable_dim)? [sep] ELLIPSIS [sep] (bare_tuple{x} | replacable_dim)?
        ?sep: ","
        ?dim: named_dim | positional_dim
        named_dim: NAME
        positional_dim: EMPTY
        NAME: /_\w*|[^_\W\d]\w*/
        EMPTY: "_"
        ELLIPSIS: "..."
    """,
    parser="earley",
    maybe_placeholders=False,
)


def parse_dims(dims_raw: Union[str, Tuple]) -> Tuple:
    """
    Parses a dimensions declaration string into a tree.

    Dimensions string example: "you_can_use_commas,or_spaces to_separate_dimensions ( l (m n) ) dimension_identifiers_can_be_any_valid_python_identifier"
    """
    if not dims_raw:
        return einexpr.array_api.dimension.DimensionTuple(())

    def helper(dims_raw):
        if isinstance(dims_raw, (tuple, list, einexpr.array_api.dimension.DimensionTuple)):
            children = []
            for child in dims_raw:
                child_processed = helper(child)
                if isinstance(child_processed, einexpr.utils.ExpandSentinel):
                    children.extend(child_processed.content)
                else:
                    children.append(child_processed)
            return einexpr.array_api.dimension.DimensionTuple(tuple(children))
        elif isinstance(dims_raw, einexpr.array_api.dimension.DimensionReplacement):
            return einexpr.array_api.dimension.DimensionReplacement(
                helper(dims_raw.original), helper(dims_raw.replacement)
            )
        elif isinstance(dims_raw, einexpr.array_api.dimension.Dimension):
            return dims_raw
        elif isinstance(dims_raw, str):
            tree = dims_reshape_grammar.parse(dims_raw)
            return TreeToReshapeDimension().transform(tree)
        elif dims_raw is None:
            return None
        else:
            raise ValueError(f"Unexpected value {dims_raw}.")
    
    dims = helper(dims_raw)
    if isinstance(dims, einexpr.utils.ExpandSentinel):
        dims = einexpr.array_api.dimension.DimensionTuple(dims.content)
    elif isinstance(dims_raw, str) or isinstance(dims, einexpr.array_api.dimension.Dimension):
        dims = einexpr.array_api.dimension.DimensionTuple((dims,))
    # Replace any ``None`` values with their positional index
    def apply_positional_index(dim, position):
        if dim is None or isinstance(dim, (einexpr.array_api.dimension.PositionalDimension, einexpr.array_api.dimension.PositionalDimensionPlaceholder)):
            match len(position):
                case 0:
                    raise einexpr.exceptions.InternalError(f"The base dimension should be a tuple, not a {dim}.")
                case 1:
                    if dim is None:
                        return einexpr.array_api.dimension.PositionalDimension(position[0] - len(dims))
                    elif isinstance(dim, einexpr.array_api.dimension.PositionalDimension):
                        return einexpr.array_api.dimension.PositionalDimension(position[0] - len(dims), size=dim.size)
                    elif isinstance(dim, einexpr.array_api.dimension.PositionalDimensionPlaceholder):
                        return einexpr.array_api.dimension.PositionalDimension(position[0] - len(dims))
                    else:
                        raise einexpr.exceptions.UnreachableError()
                case _:
                    raise einexpr.exceptions.ValueError(f"Positional dimension {position} is not a singleton. Positional dimensions can only appear at the first level of a dimension tree.")
        else:
            return dim
    dims = map_over_dims(apply_positional_index, dims, enumerate_positions=True, strict=False)
    dims = propogate_sizes(dims)
    if not isinstance(dims, einexpr.array_api.dimension.DimensionTuple):
        raise ValueError(f"Unexpected value {dims}.")
    return dims


def parse_dims_reshape(dims_raw: Union[str, Tuple]) -> Tuple:
    """
    Parses a dimensions reshape string into a tree.
    """
    return parse_dims(dims_raw)


def compute_total_size(dims: einexpr.array_api.dimension.Dimensions) -> Optional[int]:
    """
    Returns the total dimensions size, or None if the dimensions are not known.
    """
    if isinstance(dims, einexpr.array_api.dimension.Dimension):
        return dims.size
    elif isinstance(dims, einexpr.array_api.dimension.DimensionReplacement):
        return compute_total_size(dims.replacement)
    elif isinstance(dims, (tuple, list, einexpr.array_api.dimension.DimensionTuple)):
        size = 1
        for dim in dims:
            if dim.size is None:
                raise ValueError(f"Dimension {dim} has unknown size.")
            size *= dim.size
        return size
    else:
        raise ValueError(f"Unexpected value {dims}.")


def dims_to_shape(dims: D) -> Tuple[int, ...]:
    """
    Converts a dimensions object into a shape tuple.
    """
    return tuple(compute_total_size(dim) for dim in dims)


def iter_dims(dim_obj: einexpr.array_api.dimension.DimensionObject, enumerate_positions: bool = False) -> Iterator[einexpr.array_api.dimension.Dimension]:
    """
    Iterates over all dimension objects.
    """
    def _iter_dims(dim_obj, position):
        if isinstance(dim_obj, einexpr.array_api.dimension.DimensionReplacement):
            yield (dim_obj, position)
            yield from _iter_dims(dim_obj.original, position + (0,))
            yield from _iter_dims(dim_obj.replacement, position + (1,))
        elif isinstance(dim_obj, (list, set, tuple, einexpr.array_api.dimension.DimensionTuple)):
            yield (dim_obj, position)
            for i, dim in enumerate(dim_obj):
                yield from _iter_dims(dim, position + (i,))
        else:
            yield (dim_obj, position)
    yield from ((dim_obj, pos) if enumerate_positions else dim_obj for dim_obj, pos in _iter_dims(dim_obj, ()))


def map_over_dims(f: Callable[einexpr.array_api.dimension.DimensionObject, einexpr.array_api.dimension.DimensionObject], dim_obj: D, strict: bool = True, enumerate_positions: bool = False) -> D:
    """
    Maps a function over all dimension objects.
    """
    def _map_over_dims(dim_obj, position):
        if isinstance(dim_obj, einexpr.array_api.dimension.DimensionReplacement):
            return einexpr.array_api.dimension.DimensionReplacement(_map_over_dims(dim_obj.original, position + (0,)), _map_over_dims(dim_obj.replacement, position + (1,)))
        elif isinstance(dim_obj, (list, set, tuple, einexpr.array_api.dimension.DimensionTuple)):
            return type(dim_obj)(_map_over_dims(dim, position + (i,)) for i, dim in enumerate(dim_obj))
        elif isinstance(dim_obj, einexpr.array_api.dimension.Dimension):
            return f(dim_obj, position) if enumerate_positions else f(dim_obj)
        elif isinstance(dim_obj, str):
            dim_obj = einexpr.array_api.dimension.NamedDimension(dim_obj)
            return  f(dim_obj, position) if enumerate_positions else f(dim_obj)
        elif strict:
            raise ValueError(f"Unexpected value {dim_obj}.")
        else:
            return f(dim_obj, position) if enumerate_positions else f(dim_obj)
    return _map_over_dims(dim_obj, ())


def transform_dims_tree(tree: D) -> D:
    """
    Transforms a tree into a dimension object.
    """
    if isinstance(tree, einexpr.array_api.dimension.Dimension):
        return tree
    elif isinstance(tree, einexpr.array_api.dimension.DimensionReplacement):
        return einexpr.array_api.dimension.DimensionReplacement(transform_dims_tree(tree.original), transform_dims_tree(tree.replacement))
    elif isinstance(tree, (tuple, einexpr.array_api.dimension.DimensionTuple)):
        return einexpr.array_api.dimension.DimensionTuple(transform_dims_tree(dim) for dim in tree)
    else:
        raise ValueError(f"Unexpected value {tree}.")


def gather_sizes(dims: einexpr.array_api.dimension.DimensionObject) -> Dict[str, int]:
    """
    Returns a dictionary of dimension names and sizes.
    """
    sizes = {}
    for dim in iter_dims(dims):
        if isinstance(dim, einexpr.array_api.dimension.Dimension) and dim.size is not None:
            if dim.id in sizes:
                if sizes[dim.id] != dim.size:
                    raise ValueError(f"Dimension {dim.id} has conflicting sizes: {sizes[dim.id]} and {dim.size}.")
            else:
                sizes[dim.id] = dim.size
    return sizes


def gather_names(dims: einexpr.array_api.dimension.DimensionObject) -> Set[str]:
    """
    Returns a set of dimension names.
    """
    names = set()
    for dim in iter_dims(dims):
        if isinstance(dim, einexpr.array_api.dimension.Dimension):
            names.add(dim.id)
    return names


def apply_sizes(dims: D, sizes: Dict[str, int], overwrite: bool = False) -> D:
    """
    Applies a dictionary of dimension names and sizes to a dimension object, raising an error if the size is already set differently.
    """
    def helper(dims: D) -> D:
        if isinstance(dims, einexpr.array_api.dimension.Dimension) and dims.id in sizes:
            if not overwrite and dims.size is not None and dims.size != sizes[dims.id]:
                raise einexpr.exceptions.DimensionMismatch(f"Dimension {dims.id} has different sizes: {dims.size} != {sizes[dims.id]}")
            return replace(dims, size=sizes.get(dims.id, None))
        else:
            return dims
    return map_over_dims(helper, dims, strict=False)


def propogate_sizes(dims: D, sizes: Optional[Dict[str, int]] = None) -> D:
    """
    Propogates dimension sizes throughout the dimension object, raising an error when two or more named dimensions with the same name have different sizes.
    """
    # Gather all named dimension sizes
    sizes = sizes or {}
    sizes |= gather_sizes(dims)
    # Apply sizes
    return apply_sizes(dims, sizes)


def expand_ellipsis(dims: einexpr.array_api.dimension.DimensionObject, defaults: einexpr.array_api.dimension.Dimensions) -> einexpr.array_api.dimension.DimensionObject:
    """
    Expands a dimension object with ellipsis into a dimension object with the given shape.
    """
    if ... in dims:
        i_ellipsis = dims.index(...)
        before_ellipsis = dims[:i_ellipsis]
        after_ellipsis = dims[i_ellipsis + 1:]
        len_ellipsis = len(defaults) - len(before_ellipsis) - len(after_ellipsis)
        ellipsis_replacement = defaults[i_ellipsis:i_ellipsis + len_ellipsis]
        dims = (*before_ellipsis, *ellipsis_replacement, *after_ellipsis)
    return dims

def process_dims_declaration(dims_raw: Union[str, Tuple, None], shape: Tuple[int, ...]) -> einexpr.array_api.dimension.Dimensions:
    """
    Processes a dimensions declaration into a dimension object.
    """
    if dims_raw is None:
        # Use full tuple of positional dimensions represented by ``None``
        dims_raw = (None,) * len(shape)
    elif not isinstance(dims_raw, (str, tuple, list, einexpr.array_api.dimension.DimensionTuple)):
        raise ValueError(f"Unexpected value {dims_raw}.")
    dims = parse_dims(dims_raw)
    # dims = expand_ellipsis(dims, shape)
    defaults = shape
    if ... in dims:
        i_ellipsis = dims.index(...)
        before_ellipsis = dims[:i_ellipsis]
        after_ellipsis = dims[i_ellipsis + 1:]
        len_ellipsis = len(defaults) - len(before_ellipsis) - len(after_ellipsis)
        ellipsis_replacement = defaults[i_ellipsis:i_ellipsis + len_ellipsis]
        dims = (*before_ellipsis, *ellipsis_replacement, *after_ellipsis)
    if len(dims) != len(shape):
        raise ValueError(f"Declaration {dims_raw} has wrong number of dimensions. Expected {len(shape)}, got {len(dims)}.")
    dim_sizes = gather_sizes(dims) | {dim.id: size for dim, size in zip(dims, shape) if isinstance(dim, einexpr.array_api.dimension.Dimension)}
    dims = propogate_sizes(dims, dim_sizes)
    if any(compute_total_size(dim) is None for dim in dims):
        raise einexpr.exceptions.InternalError("All dimensions must have a size.")
    return dims


def process_dims_reshape(dims_raw: Union[str, Tuple], existing_dims: einexpr.array_api.dimension.DimensionObject) -> einexpr.array_api.dimension.Dimensions:
    """
    Processes a dimensions reshape string into a dimension object.
    """
    if not dims_raw:
        return einexpr.array_api.dimension.DimensionTuple()
    dims = parse_dims_reshape(dims_raw)
    dims = propogate_sizes(dims, gather_sizes(existing_dims))
    existing_dims = propogate_sizes(existing_dims, gather_sizes(dims))
    # dims = expand_ellipsis(dims, existing_dims)
    if ... in dims:
        i_ellipsis = dims.index(...)
        before_ellipsis = dims[:i_ellipsis]
        after_ellipsis = dims[i_ellipsis + 1:]
        in_ellipsis = set(existing_dims) - set(before_ellipsis) - set(after_ellipsis)
        ellipsis_replacement = [dim for dim in existing_dims if dim in in_ellipsis]
        dims = (*before_ellipsis, *ellipsis_replacement, *after_ellipsis)
    # Bind positional dimensions
    replacements = get_positional_dim_replacements((dims, existing_dims))
    replacees = {named_dim: pos_dim for pos_dim, named_dim in replacements.items()}
    dims = map_over_dims(lambda dim: einexpr.array_api.dimension.DimensionReplacement(replacees[dim], dim) if dim in replacees else dim, dims)
    return dims


def dims_equivalent(dims1: einexpr.array_api.dimension.Dimensions, dims2: einexpr.array_api.dimension.Dimensions, ignore_missing_sizes: bool = True) -> bool:
    """
    Returns True if the two dimension objects are equivalent.
    """
    if isinstance(dims1, einexpr.array_api.dimension.DimensionReplacement):
        return dims_equivalent(dims1.original, dims2, ignore_missing_sizes) and dims_equivalent(dims1.replacement, dims2, ignore_missing_sizes)
    elif isinstance(dims1, (tuple, einexpr.array_api.dimension.DimensionTuple)):
        if isinstance(dims2, (tuple, einexpr.array_api.dimension.DimensionTuple)):
            if len(dims1) != len(dims2):
                return False
            for dim1, dim2 in zip(dims1, dims2):
                if not dims_equivalent(dim1, dim2, ignore_missing_sizes):
                    return False
            return True
        else:
            return False
    elif isinstance(dims1, einexpr.array_api.dimension.Dimension):
        if isinstance(dims2, einexpr.array_api.dimension.Dimension):
            if dims1.id != dims2.id:
                return False
            if dims1.size is None or dims2.size is None:
                return True
            return dims1.size == dims2.size
        else:
            return False
    else:
        raise ValueError(f"Unexpected value {dims1}.")


def dims_issubset(dims1: einexpr.array_api.dimension.DimensionTuple, dims2: einexpr.array_api.dimension.DimensionTuple, ignore_missing_sizes: bool = True) -> bool:
    """
    Returns True if the first tuple of dimensions is a subset of the second tuple of dimensions.
    """
    # We need to fill in the missing sizes in each tuple. If we don't, we could get a named dimension with a missing size (i.e. size=None) 'mismatching' against a dimension with the same name but a defined size.
    # Gather sizes
    sizes = gather_sizes(dims1) | gather_sizes(dims2)
    # Apply sizes
    try:
        dims1 = apply_sizes(dims1, sizes)
        dims2 = apply_sizes(dims2, sizes)
    except ValueError:
        return False
    # Check if dims1 is a subset of dims2
    return set(dims1) <= set(dims2)


def primitivize_dims(dims: einexpr.array_api.dimension.DimensionObject) -> einexpr.array_api.dimension.DimensionObject:
    """
    Converts a dimensions object into simple (possibly nested) tuples of strings. This is a lossy conversion.
    """
    if isinstance(dims, einexpr.array_api.dimension.DimensionReplacement):
        return primitivize_dims(dims.replacement)
    elif isinstance(dims, (list, tuple, einexpr.array_api.dimension.DimensionTuple)):
        return tuple(primitivize_dims(dim) for dim in dims)
    elif isinstance(dims, einexpr.array_api.dimension.NamedDimension):
        return dims.id
    elif isinstance(dims, einexpr.array_api.dimension.PositionalDimension):
        return dims.position
    elif isinstance(dims, (int, str)):
        return dims
    else:
        raise ValueError(f"Unexpected value {dims}.")


def ignore_replacements(dims: D) -> D:
    """
    Replaces all dimension replacements with their original dimensions.
    """
    if isinstance(dims, einexpr.array_api.dimension.DimensionReplacement):
        return dims.original
    elif isinstance(dims, (tuple, einexpr.array_api.dimension.DimensionTuple)):
        return einexpr.array_api.dimension.DimensionTuple(ignore_replacements(dim) for dim in dims)
    elif isinstance(dims, einexpr.array_api.dimension.Dimension):
        return dims
    else:
        raise ValueError(f"Unexpected value {dims}.")


def apply_replacements(dims: D) -> D:
    """
    Replaces all dimension replacements with their replacement dimensions.
    """
    if isinstance(dims, einexpr.array_api.dimension.DimensionReplacement):
        return dims.replacement
    elif isinstance(dims, (tuple, einexpr.array_api.dimension.DimensionTuple)):
        return einexpr.array_api.dimension.DimensionTuple(apply_replacements(dim) for dim in dims)
    elif isinstance(dims, einexpr.array_api.dimension.Dimension):
        return dims
    else:
        raise ValueError(f"Unexpected value {dims}.")


def replace_dims(array: einexpr.array_api.einarray, replacements: Dict[einexpr.array_api.dimension.DimensionObject, einexpr.array_api.dimension.DimensionObject]) -> Tuple[einexpr.array_api.dimension.DimensionObject, ...]:
    """
    Applies a dictionary of replacements to a tuple of dimensions.
    """
    return einexpr.einarray(
        array.a,
        dims=map_over_dims(lambda dim: replacements.get(dim, dim), array.dims),
        ambiguous_dims=map_over_dims(lambda dim: replacements.get(dim, dim), array.ambiguous_dims),
    )


def expand_dims(dims: D) -> D:
    """
    Expands all composite dimensions into their constituents, effectually 'flattening' the dimension tree itself.
    """
    def helper(dims: D) -> D:
        if isinstance(dims, einexpr.array_api.dimension.DimensionReplacement):
            raise ValueError(f"Can't expand a dimension replacement: {dims}. Use ignore_replacements() or apply_replacements() first.")
        elif isinstance(dims, (tuple, einexpr.array_api.dimension.DimensionTuple)):
            for dim in dims:
                yield from helper(dim)
        elif isinstance(dims, (str, einexpr.array_api.dimension.Dimension)):
            yield dims
        else:
            raise ValueError(f"Unexpected value {dims}.")
    return tuple(helper(dims))


def isolate_dim(array_dims: Tuple[einexpr.array_api.Dimension], dim: einexpr.array_api.Dimension) -> Tuple[einexpr.array_api.Dimension, einexpr.array_api.Dimension]:
    """
    Extract the dimensions in ``dim`` into the first level if it is part of a composite dimension.
    """    
    position = einexpr.utils.get_position(array_dims, dim)
    if position is None:
        raise ValueError(f"Dimension {dim} is not part of the array dimensions {array_dims}.")
    lhs, rhs = einexpr.utils.split_at(array_dims, position)
    return einexpr.utils.deep_remove((*lhs, dim, *rhs), lambda x: x == ())


def isolate_dims(array_dims: Tuple[einexpr.array_api.Dimension], dims: Iterable[einexpr.array_api.Dimension]) -> einexpr.array_api.einarray:
    """
    Extract the dimensions in ``dims`` into the first level if they are part of a composite dimension.
    """
    array_dims = einexpr.dimension_utils.primitivize_dims(array_dims)
    dims = einexpr.dimension_utils.primitivize_dims(dims)
    for dim in dims:
        array_dims = isolate_dim(array_dims, dim)
    return array_dims


def get_positional_dim_replacements(dimss: Tuple[einexpr.array_api.dimension.DimensionObject, ...]) -> Dict[einexpr.array_api.dimension.PositionalDimension, einexpr.array_api.dimension.NamedDimension]:
    """
    Infer the names of positional arguments.
    """
    positional_dims = {}
    for dims in dimss:
        for dim in iter_dims(dims):
            if isinstance(dim, einexpr.array_api.dimension.PositionalDimension):
                if dim.id in positional_dims:
                    if positional_dims[dim.id] != dim:
                        raise ValueError(f"Positional dimensions conflict: {positional_dims[dim.id]!r} and {dim!r}.")
                else:
                    positional_dims[dim.id] = dim
    replacements = {}
    for dim in positional_dims.values():
        for dims in dimss:
            if len(dims) >= -dim.position:
                if dim in replacements:
                    if type(replacements[dim]) is type(dims[dim.position]) and replacements[dim] != dims[dim.position]:
                        raise ValueError(f"Positional dimension conflict: {replacements[dim]!r} and {dims[dim.position]!r}.")
                elif isinstance(dims[dim.position], einexpr.array_api.dimension.NamedDimension):
                    if dim.size is not None and dims[dim.position].size is not None:
                        if dim.size != dims[dim.position].size:
                            raise ValueError(f"Dimension sizes conflict: {dim!r} and {dims[dim.position]!r}.")
                        replacements[dim] = dims[dim.position]
                    elif dims[dim.position].size is None:
                        replacements[dim] = dims[dim.position].with_size(dim.size)
                    else:
                        replacements[dim] = dims[dim.position]
                    replacements[dim] = dims[dim.position]
                elif isinstance(dims[dim.position], (list, tuple, einexpr.array_api.dimension.DimensionTuple)):
                    pos_dim_size = compute_total_size(dim)
                    replacement_dim_size = compute_total_size(dims[dim.position])
                    if pos_dim_size is not None and replacement_dim_size is not None:
                        if pos_dim_size != replacement_dim_size:
                            raise ValueError(f"Dimension sizes conflict: {dim!r} and {dims[dim.position]!r}.")
                    replacements[dim] = dims[dim.position]
                elif isinstance(dims[dim.position], einexpr.array_api.dimension.PositionalDimension):
                    pass
                else:
                    raise ValueError(f"Unexpected value {dims[dim.position]!r}.")
    return replacements


def is_dimensionless(array):
    return isinstance(array, (int, float)) or einexpr.backends.conforms_to_array_api(array) and len(array.shape) == 0


def get_dims(array):
    if isinstance(array, einexpr.array):
        return array.dims
    elif is_dimensionless(array):
        return ()
    else:
        raise TypeError(f'{array} is not a recognized einexpr array')


def resolve_positional_dims(arrays: Tuple[einexpr.array_api.einarray, ...]) -> Tuple[einexpr.array_api.einarray, ...]:
    """
    Resolve positional dimensions.
    """
    dimss = tuple(get_dims(array) for array in arrays)
    replacements = get_positional_dim_replacements(dimss)
    return tuple(replace_dims(array, replacements) if isinstance(array, einexpr.array_api.einarray) else array for array in arrays)