import collections
import functools
import operator
from os import PRIO_PGRP
import re
from itertools import chain, zip_longest
from types import EllipsisType
from typing import Any, Callable, Dict, Iterable, Iterator, List, Literal, Optional, Sequence, Set, Tuple, TypeVar, Union

import numpy as np
from pydantic import PositiveInt
from einexpr.array_api.dimension import NamedDimension, PositionalDimension

from einexpr.utils.utils import pedantic_dict_merge
from .exceptions import AmbiguousDimensionException
import einexpr
from lark import Lark, Transformer, v_args

from dataclasses import replace


D = TypeVar("D", bound=einexpr.array_api.dimension.DimensionObject)


def _convert_from_spec(dims):
    if isinstance(dims, einexpr.array_api.dimension.DimensionSpecification):
        return dims.dimensions
    return dims



def get_final_aligned_dims(
    *ein_dims: List[einexpr.array_api.dimension.AtomicDimension],
) -> List[einexpr.array_api.dimension.AtomicDimension]:
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
    *dims: List[einexpr.array_api.dimension.AtomicDimension],
) -> Set[einexpr.array_api.dimension.AtomicDimension]:
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
    *ein_dims: List[einexpr.array_api.dimension.AtomicDimension],
) -> List[List[einexpr.array_api.dimension.AtomicDimension]]:
    """
    Returns a list of lists of dimensions that are aligned with eachother.
    """
    aligned_final = get_final_aligned_dims(*ein_dims)
    return [[dim for dim in aligned_final if dim in dims] for dims in ein_dims]


def calculate_transexpand(
    dims_from: List[einexpr.array_api.dimension.AtomicDimension],
    dims_to: List[einexpr.array_api.dimension.AtomicDimension],
) -> List[einexpr.array_api.dimension.AtomicDimension]:
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

    def replacable_dim(self, value) -> Union[einexpr.array_api.dimension.AtomicDimension, einexpr.array_api.dimension.DimensionTuple]:
        return value
        
    def dim_replacement(self, original, replacement) -> einexpr.array_api.dimension.DimensionReplacement:
        return einexpr.array_api.dimension.DimensionReplacement(original, replacement)

    def irreplacable_dim(self, value) -> Union[einexpr.array_api.dimension.AtomicDimension, einexpr.array_api.dimension.DimensionTuple]:
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
        return einexpr.array_api.dimension.AbsorbableDimension()

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
        NAME: /_\w+|[^_\W\d]\w*/
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
        elif isinstance(dims_raw, einexpr.utils.ExpandSentinel):
            return einexpr.utils.ExpandSentinel(tuple(helper(dim) for dim in dims_raw))
        elif isinstance(dims_raw, einexpr.array_api.dimension.DimensionReplacement):
            return einexpr.array_api.dimension.DimensionReplacement(
                helper(dims_raw.original), helper(dims_raw.replacement)
            )
        elif isinstance(dims_raw, einexpr.array_api.dimension.AtomicDimension):
            return dims_raw
        elif dims_raw is Ellipsis:
            return dims_raw
        elif isinstance(dims_raw, str):
            tree = dims_reshape_grammar.parse(dims_raw)
            return TreeToReshapeDimension().transform(tree)
        elif isinstance(dims_raw, int):
            return einexpr.array_api.dimension.PositionalDimension(dims_raw)
        elif dims_raw is None:
            return einexpr.array_api.dimension.AbsorbableDimension()
        else:
            raise ValueError(f"Unexpected value {dims_raw}.")
    
    dims = helper(dims_raw)
    if isinstance(dims, einexpr.utils.ExpandSentinel):
        dims = einexpr.array_api.dimension.DimensionTuple(dims.content)
    elif isinstance(dims_raw, str) or isinstance(dims, einexpr.array_api.dimension.AtomicDimension):
        dims = einexpr.array_api.dimension.DimensionTuple((dims,))
    dims = propogate_sizes(dims)
    dims = helper(dims)
    if not isinstance(dims, einexpr.array_api.dimension.DimensionTuple):
        raise ValueError(f"Unexpected value {dims}.")
    return dims


def parse_dims_reshape(dims_raw: Union[str, Tuple]) -> Tuple:
    """
    Parses a dimensions reshape string into a tree.
    """
    return parse_dims(dims_raw)


def compute_total_size(dims: einexpr.array_api.dimension.DimensionSpecification) -> Optional[int]:
    """
    Returns the total dimensions size, or None if the dimensions are not known.
    """
    if isinstance(dims, einexpr.array_api.dimension.AtomicDimension):
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
    dims = _convert_from_spec(dims)
    sizes = tuple(tuple(compute_total_size(dim) for dim in dims))
    assert all(size is not None for size in sizes)
    return sizes


def map_over_dims(
    f: Callable[einexpr.array_api.dimension.DimensionObject, einexpr.array_api.dimension.DimensionObject],
    dim_obj: D, strict: bool = True,
    enumerate_positions: bool = False,
    enumerate_dimension_replacement_path_mode: Literal["both", "original", "replacement", "neither"] = "both",
    enumerate_dimension_replacement_has_position: bool = False
)-> D:
    """
    Maps a function over all dimension objects.
    """
    def _map_over_dims(dim_obj, position):
        if isinstance(dim_obj, einexpr.array_api.dimension.DimensionReplacement):
            original = dim_obj.original
            replacement = dim_obj.replacement
            if enumerate_dimension_replacement_has_position:
                position_original = position + (0,)
                position_replacement = position + (1,)
            else:
                position_original = position
                position_replacement = position
            match enumerate_dimension_replacement_path_mode:
                case "both":
                    original = _map_over_dims(original, position_original)
                    replacement = _map_over_dims(replacement, position_replacement)
                case "original":
                    original = _map_over_dims(original, position_original)
                case "replacement":
                    replacement = _map_over_dims(replacement, position_replacement)
                case "neither":
                    pass
                case _:
                    raise ValueError(f"Unexpected value for argument enumerate_dimension_replacement_mode: {enumerate_dimension_replacement_path_mode}. Expecting one of 'both', 'original', 'replacement', or 'passthrough'.")
            return einexpr.array_api.dimension.DimensionReplacement(original, replacement)
        elif isinstance(dim_obj, (list, set, tuple, einexpr.array_api.dimension.DimensionTuple)):
            return type(dim_obj)(_map_over_dims(dim, position + (i,)) for i, dim in enumerate(dim_obj))
        elif isinstance(dim_obj, einexpr.array_api.dimension.AtomicDimension):
            return f(dim_obj, position) if enumerate_positions else f(dim_obj)
        elif isinstance(dim_obj, str):
            dim_obj = einexpr.array_api.dimension.NamedDimension(dim_obj)
            return  f(dim_obj, position) if enumerate_positions else f(dim_obj)
        elif isinstance(dim_obj, einexpr.array_api.dimension.DimensionSpecification):
            return einexpr.array_api.dimension.DimensionSpecification(tuple(_map_over_dims(dim, position + (i,)) for i, dim in enumerate(dim_obj)), sizes=dim_obj.sizes)
        elif dim_obj is ...:
            return dim_obj
        elif strict:
            raise ValueError(f"Unexpected value {dim_obj}.")
        else:
            return f(dim_obj, position) if enumerate_positions else f(dim_obj)
    return _map_over_dims(dim_obj, ())


def iter_dims(
    dim_obj: einexpr.array_api.dimension.DimensionObject,
    enumerate_positions: bool = False,
    enumerate_dimension_replacement_path_mode: Literal["both", "original", "replacement", "neither"] = "both",
    enumerate_dimension_replacement_has_position: bool = False
) -> Iterator[einexpr.array_api.dimension.AtomicDimension]:
    """
    Iterates over all dimension objects.
    """
    def _iter_dims(dim_obj, position):
        if isinstance(dim_obj, einexpr.array_api.dimension.DimensionReplacement):
            if enumerate_dimension_replacement_has_position:
                position_original = position + (0,)
                position_replacement = position + (1,)
            else:
                position_original = position
                position_replacement = position
            yield (dim_obj, position)
            match enumerate_dimension_replacement_path_mode:
                case "both":
                    yield from _iter_dims(dim_obj.original, position_original)
                    yield from _iter_dims(dim_obj.replacement, position_replacement)
                case "original":
                    yield from _iter_dims(dim_obj.original, position_original)
                case "replacement":
                    yield from _iter_dims(dim_obj.replacement, position_replacement)
                case "neither":
                    pass
                case _:
                    raise ValueError(f"Unexpected value for argument enumerate_dimension_replacement_mode: {enumerate_dimension_replacement_path_mode}. Expecting one of 'both', 'original', 'replacement', or 'passthrough'.")
        elif isinstance(dim_obj, (list, set, tuple, einexpr.array_api.dimension.DimensionTuple, einexpr.array_api.dimension.DimensionSpecification)):
            yield (dim_obj, position)
            for i, dim in enumerate(dim_obj):
                yield from _iter_dims(dim, position + (i,))
        else:
            yield (dim_obj, position)
    yield from ((dim_obj, pos) if enumerate_positions else dim_obj for dim_obj, pos in _iter_dims(dim_obj, ()))


def gather_sizes(dims: einexpr.array_api.dimension.DimensionObject) -> Dict[str, int]:
    """
    Returns a dictionary of dimension names and sizes.
    """
    sizes = {}
    for dim in iter_dims(dims):
        if isinstance(dim, einexpr.array_api.dimension.AtomicDimension) and dim.size is not None:
            if dim.id in sizes:
                if sizes[dim.id] != dim.size:
                    raise ValueError(f"Dimension {dim.id} has conflicting sizes: {sizes[dim.id]} and {dim.size}.")
            else:
                sizes[dim.id] = dim.size
    return sizes


def apply_sizes(dims: D, sizes: Dict[str, int], overwrite: bool = False) -> D:
    """
    Applies a dictionary of dimension names and sizes to a dimension object, raising an error if the size is already set differently.
    """
    def helper(dims: D) -> D:
        if isinstance(dims, einexpr.array_api.dimension.AtomicDimension) and dims.id in sizes:
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


def process_dims_declaration(dims_raw: Union[str, Tuple, None], shape: Tuple[int, ...]) -> einexpr.array_api.dimension.DimensionSpecification:
    """
    Processes a dimensions declaration into a dimension object.
    """
    dims_raw = _convert_from_spec(dims_raw)
    if dims_raw is None:
        # Use full tuple of positional dimensions
        dims_raw = tuple(einexpr.array_api.dimension.PositionalDimension(i, size) for i, size in enumerate(shape, start=-len(shape)))
    elif not isinstance(dims_raw, (str, tuple, list, einexpr.array_api.dimension.DimensionTuple)):
        raise ValueError(f"Unexpected value {dims_raw}.")
    dims = parse_dims(dims_raw)
    dim_sizes = gather_sizes(dims) | {dim.id: size for dim, size in zip(dims, shape) if isinstance(dim, einexpr.array_api.dimension.AtomicDimension)}
    dims = propogate_sizes(dims, dim_sizes)
    # Expand ellipsis
    if ... in dims:
        i_ellipsis = dims.index(...)
        before_ellipsis = dims[:i_ellipsis]
        after_ellipsis = dims[i_ellipsis + 1:]
        len_ellipsis = len(shape) - len(before_ellipsis) - len(after_ellipsis)
        ellipsis_replacement = (einexpr.array_api.dimension.AbsorbableDimension() for _ in range(len_ellipsis))
        dims = (*before_ellipsis, *ellipsis_replacement, *after_ellipsis)
        if len(ellipsis_replacement) > 0 and len(dims) >= len(shape):
            raise einexpr.exceptions.InternalError("Ellipsis appears to be expanding as one or more dimensions, but the size of the expanded dimensions tuple is greater than that of the shape tuple.")
    if len(dims) != len(shape):
        raise einexpr.exceptions.InternalError("DimensionSpecification tuple size does not match shape tuple size.")
    dims = tuple(einexpr.array_api.dimension.PositionalDimension(i, size) if isinstance(dim, einexpr.array_api.dimension.AbsorbableDimension) else dim for i, (dim, size) in enumerate(zip(dims, shape), start=-len(shape)))
    if len(dims) != len(shape):
        raise ValueError(f"Declaration {dims_raw} has wrong number of dimensions. Expected {len(shape)}, got {len(dims)}.")
    if any(compute_total_size(dim) is None for dim in dims):
        raise einexpr.exceptions.InternalError("All dimensions must have a size.")
    return dims


def process_dims_reshape(dims_raw: Union[str, Tuple], existing_dims: einexpr.array_api.dimension.DimensionObject) -> einexpr.array_api.dimension.DimensionSpecification:
    """
    Processes a dimensions reshape string into a dimension object.
    """
    if not dims_raw:
        return einexpr.array_api.dimension.DimensionTuple()
    dims = parse_dims_reshape(dims_raw)
    dims = propogate_sizes(dims, gather_sizes(existing_dims))
    existing_dims = propogate_sizes(existing_dims, gather_sizes(dims))
    concrete_dims = {dim for dim in ignore_replacements(dims) if isinstance(dim, (
        einexpr.array_api.dimension.NamedDimension,
        einexpr.array_api.dimension.DimensionTuple,
        tuple,
    ))}
    # NOTE: to keeps things simple for now, we don't allow complex reshaping (involving the modification of compound dimensions) when an ellipsis is present.
    #       This should be fixed in the future.
    if ... in dims:
        raise einexpr.exceptions.InternalError("Ellipsis is not yet supported.")
        if not concrete_dims <= set(existing_dims):
            raise einexpr.exceptions.InternalError(f"einexpr does not yet support reshaping to a dimension that in the set of the existing dimensions. Existing dimensions: {existing_dims}, reshape dimensions: {dims}")
        # Replace ellipsis with absorbing dimensions for the remaining dimensions
        i_ellipsis = dims.index(...)
        dims = (
            *dims[:i_ellipsis],
            *(einexpr.array_api.dimension.AbsorbableDimension() for _ in range(i_ellipsis, len(existing_dims) - (len(dims) - i_ellipsis - 1))),
            *dims[i_ellipsis + 1:]
        )
        assert len(dims) == len(existing_dims)
    dims = propogate_sizes(dims, gather_sizes(existing_dims))
    # Bind positional dimensions
    replacements = get_positional_dim_replacements((ignore_replacements(dims), existing_dims))
    replacees = {named_dim: pos_dim for pos_dim, named_dim in replacements.items()}
    dims = map_over_dims((lambda dim: einexpr.array_api.dimension.DimensionReplacement(replacees[dim], dim) if dim in replacees else dim), dims, enumerate_dimension_replacement_path_mode='original')
    assert not any(isinstance(dim, einexpr.array_api.dimension.AbsorbableDimension) for dim in iter_dims(dims, enumerate_dimension_replacement_has_position="both"))
    return dims


def primitivize_dims(dims: einexpr.array_api.dimension.DimensionObject) -> einexpr.array_api.dimension.DimensionObject:
    """
    Converts a dimensions object into simple (possibly nested) tuples of strings. This is a lossy conversion.
    """
    dims = _convert_from_spec(dims)
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
        return ignore_replacements(dims.original)
    elif isinstance(dims, (tuple, einexpr.array_api.dimension.DimensionTuple)):
        return einexpr.array_api.dimension.DimensionTuple(ignore_replacements(dim) for dim in dims)
    elif isinstance(dims, einexpr.array_api.dimension.AtomicDimension):
        return dims
    elif dims is ...:
        return dims
    else:
        raise ValueError(f"Unexpected value {dims}.")


def apply_replacements(dims: D) -> D:
    """
    Replaces all dimension replacements with their replacement dimensions.
    """
    if isinstance(dims, einexpr.array_api.dimension.DimensionReplacement):
        return apply_replacements(dims.replacement)
    elif isinstance(dims, (tuple, einexpr.array_api.dimension.DimensionTuple)):
        return einexpr.array_api.dimension.DimensionTuple(apply_replacements(dim) for dim in dims)
    elif isinstance(dims, einexpr.array_api.dimension.AtomicDimension):
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
    dims = _convert_from_spec(dims)
    def helper(dims: D) -> D:
        if isinstance(dims, einexpr.array_api.dimension.DimensionReplacement):
            raise ValueError(f"Can't expand a dimension replacement: {dims}. Use ignore_replacements() or apply_replacements() first.")
        elif isinstance(dims, (tuple, einexpr.array_api.dimension.DimensionTuple)):
            for dim in dims:
                yield from helper(dim)
        elif isinstance(dims, (int, str, einexpr.array_api.dimension.AtomicDimension)):
            yield dims
        else:
            raise ValueError(f"Unexpected value {dims}.")
    return tuple(helper(dims))


def isolate_dim(array_dims: Tuple[einexpr.array_api.AtomicDimension], dim: einexpr.array_api.AtomicDimension) -> Tuple[einexpr.array_api.AtomicDimension, einexpr.array_api.AtomicDimension]:
    """
    Extract the dimensions in ``dim`` into the first level if it is part of a composite dimension.
    """    
    position = einexpr.utils.get_position(array_dims, dim)
    if position is None:
        raise ValueError(f"Dimension {dim} is not part of the array dimensions {array_dims}.")
    lhs, rhs = einexpr.utils.split_at(array_dims, position)
    return einexpr.utils.deep_remove((*lhs, dim, *rhs), lambda x: x == ())


def isolate_dims(array_dims: Tuple[einexpr.array_api.AtomicDimension], dims: Iterable[einexpr.array_api.AtomicDimension]) -> einexpr.array_api.einarray:
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
    # Replace all absorbable dimensions by positional dimensions.
    positional_to_absorbable = {}
    _dimss = []
    for dims in dimss:
        _dims = []
        for i, dim in enumerate(dims, start=-len(dims)):
            if isinstance(dim, einexpr.array_api.dimension.AbsorbableDimension):
                pos_dim = einexpr.array_api.dimension.PositionalDimension(i)
                positional_to_absorbable[pos_dim] = dim
                _dims.append(pos_dim)
            else:
                _dims.append(dim)
        _dimss.append(_dims)
    dimss = _dimss
    positional_dims = {}
    for dims in dimss:
        for i, dim in enumerate(dims, start=-len(dims)):
            if isinstance(dim, (einexpr.array_api.dimension.PositionalDimension)):
                if isinstance(dim, einexpr.array_api.dimension.PositionalDimension) and dim.position != i:
                    raise ValueError(f"Positional dimension {dim} has position {dim.position} but should have position {i}.")
                if i in positional_dims:
                    if positional_dims[dim.position] != dim:
                        raise ValueError(f"Positional or Absorbable dimensions conflict: {positional_dims[i]!r} and {dim!r}.")
                else:
                    positional_dims[i] = dim
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
                elif isinstance(dims[dim.position], (einexpr.array_api.dimension.PositionalDimension, einexpr.array_api.dimension.AbsorbableDimension)):
                    pass
                else:
                    raise ValueError(f"Unexpected value {dims[dim.position]!r}.")
    # Reverse the replacement of positional dimensions that we did at the start of the function.
    for positional, absorbable in positional_to_absorbable.items():
        if positional in replacements:
            replacements[absorbable] = replacements[positional]
            del replacements[positional]
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