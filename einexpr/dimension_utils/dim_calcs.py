import functools
import operator
from os import PRIO_PGRP
import re
from itertools import chain, zip_longest
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, TypeVar, Union

import numpy as np
from pydantic import PositiveInt
from einexpr.array_api.dimension import NamedDimension

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
    scs = einexpr.utils.get_all_scs_with_nonunique_elems_removed(*ein_dims)
    if len(scs) == 0:
        raise ValueError("No valid alignment found.")
    return sorted(scs).pop()


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

    def root_dims(self, *dims) -> einexpr.array_api.dimension.DimensionTuple:
        return einexpr.array_api.dimension.DimensionTuple(dim for dim in dims if dim is not None)

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

    def tuple(self, *dims) -> einexpr.array_api.dimension.DimensionTuple:
        return einexpr.array_api.dimension.DimensionTuple(dim for dim in dims if dim is not None)
    
    def sequence(self, *xs) -> Tuple:
        return einexpr.array_api.dimension.DimensionTuple(x for x in xs if x is not None)

    def sep(self) -> None:
        return None
    
    def dim(self, value) -> einexpr.array_api.dimension.NamedDimension:
        return einexpr.array_api.dimension.NamedDimension(value)

    def NAME(self, tree) -> einexpr.array_api.dimension.Dimension:
        return tree.value


dims_reshape_grammar = Lark(
    r"""
        %import common.WS

        %ignore WS

        start: root_dims
        ?root_dims: (replacable_dim [sep])* | dim_replacement
        ?replacable_dim: dim | "(" ( replacable_dim | dim_replacement ) ")" | tuple{replacable_dim}
        dim_replacement: ( irreplacable_dim | sequence{irreplacable_dim} ) "->" ( irreplacable_dim | sequence{replacable_dim} )
        ?irreplacable_dim: dim | "(" irreplacable_dim ")" | tuple{irreplacable_dim}
        tuple{x}: "(" x sep ")" | "(" sequence{x} ")"
        sequence{x}: x [sep] (x [sep])+
        ?sep: ","
        dim: NAME
        NAME: /[^\W\d]\w*/
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
        return ()

    def helper(dims_raw):
        if isinstance(dims_raw, (tuple, list, einexpr.array_api.dimension.DimensionTuple)):
            return einexpr.array_api.dimension.DimensionTuple((helper(dim) for dim in dims_raw), size=compute_total_size(dims_raw))
        elif isinstance(dims_raw, str):
            tree = dims_reshape_grammar.parse(dims_raw)
            return TreeToReshapeDimension().transform(tree)
        else:
            raise ValueError(f"Unexpected value {dims_raw}.")
    dims = helper(dims_raw)
    dims = propogate_sizes(dims)
    if isinstance(dims, einexpr.array_api.dimension.Dimension):
        return (dims,)
    else:
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
                return None
            size *= dim.size
        return size
    else:
        raise ValueError(f"Unexpected value {dims}.")


def iter_dims(dims: einexpr.array_api.dimension.DimensionObject) -> Iterator[einexpr.array_api.dimension.Dimension]:
    """
    Iterates over all dimension objects.
    """
    if isinstance(dims, einexpr.array_api.dimension.DimensionReplacement):
        yield dims
        yield from iter_dims(dims.original)
        yield from iter_dims(dims.replacement)
    elif isinstance(dims, (tuple, einexpr.array_api.dimension.DimensionTuple)):
        yield dims
        for dim in dims:
            yield from iter_dims(dim)
    elif isinstance(dims, (list, set)):
        yield dims
        for dim in dims:
            yield from iter_dims(dim)
    elif isinstance(dims, einexpr.array_api.dimension.Dimension):
        yield dims
    else:
        raise ValueError(f"Unexpected value {dims}.")


def map_over_dims(f: Callable[einexpr.array_api.dimension.DimensionObject, einexpr.array_api.dimension.DimensionObject], dims: D) -> D:
    """
    Maps a function over all dimension objects.
    """
    if isinstance(dims, einexpr.array_api.dimension.DimensionReplacement):
        return einexpr.array_api.dimension.DimensionReplacement(map_over_dims(f, dims.original), map_over_dims(f, dims.replacement))
    elif isinstance(dims, (tuple, einexpr.array_api.dimension.DimensionTuple)):
        return einexpr.array_api.dimension.DimensionTuple(map_over_dims(f, dim) for dim in dims)
    elif isinstance(dims, (tuple, list, set)):
        return type(dims)(map_over_dims(f, dim) for dim in dims)
    elif isinstance(dims, einexpr.array_api.dimension.Dimension):
        return f(dims)
    else:
        raise ValueError(f"Unexpected value {dims}.")


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
        if isinstance(dim, einexpr.array_api.dimension.NamedDimension) and dim.size is not None:
            if dim.name in sizes:
                if sizes[dim.name] != dim.size:
                    raise ValueError(f"Dimension {dim.name} has different sizes: {sizes[dim.name]} != {dim.size}")
            else:
                sizes[dim.name] = dim.size
    return sizes


def gather_names(dims: einexpr.array_api.dimension.DimensionObject) -> Set[str]:
    """
    Returns a set of dimension names.
    """
    names = set()
    for dim in iter_dims(dims):
        if isinstance(dim, einexpr.array_api.dimension.NamedDimension):
            names.add(dim.name)
    return names


def apply_sizes(dims: D, sizes: Dict[str, int], overwrite: bool = False) -> D:
    """
    Applies a dictionary of dimension names and sizes to a dimension object, raising an error if the size is already set differently.
    """
    def helper(dims: D) -> D:
        if isinstance(dims, einexpr.array_api.dimension.NamedDimension) and dims.name in sizes:
            if not overwrite and dims.size is not None and dims.size != sizes[dims.name]:
                raise einexpr.exceptions.DimensionMismatch(f"Dimension {dims.name} has different sizes: {dims.size} != {sizes[dims.name]}")
            return replace(dims, size=sizes.get(dims.name, None))
        else:
            return dims
    return map_over_dims(helper, dims)


def propogate_sizes(dims: D, sizes: Optional[Dict[str, int]] = None) -> D:
    """
    Propogates dimension sizes throughout the dimension object, raising an error when two or more named dimensions with the same name have different sizes.
    """
    # Gather all named dimension sizes
    sizes = sizes or {}
    sizes |= gather_sizes(dims)
    # Apply sizes
    return apply_sizes(dims, sizes)


def process_dims_declaration(dims_raw: Union[str, Tuple], shape: Tuple[int]) -> einexpr.array_api.dimension.Dimensions:
    """
    Processes a dimensions declaration string into a dimension object.
    """
    if not dims_raw:
        return einexpr.array_api.dimension.DimensionTuple()
    if len(dims_raw) != len(shape):
        raise ValueError(f"Dimensions declaration has wrong number of dimensions: {dims_raw}")
    dims = parse_dims(dims_raw)
    sizes = {dim.name: size for dim, size in zip(dims, shape)}
    return propogate_sizes(dims, sizes)


def process_dims_reshape(dims_raw: Union[str, Tuple], shape: Tuple[int]) -> einexpr.array_api.dimension.Dimensions:
    """
    Processes a dimensions reshape string into a dimension object.
    """
    if not dims_raw:
        return ()
    dims = parse_dims_reshape(dims_raw)
    sizes = {dim.name: size for dim, size in zip(dims, shape)}
    return propogate_sizes(dims, sizes)


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
            if dims1.name != dims2.name:
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