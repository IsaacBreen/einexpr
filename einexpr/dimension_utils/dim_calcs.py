import collections
import functools
import itertools
import operator
from os import PRIO_PGRP
import re
from itertools import chain, repeat, zip_longest
from types import EllipsisType
from typing import Any, Callable, Dict, Iterable, Iterator, List, Literal, Optional, Sequence, Set, Tuple, TypeVar, Union, overload

import numpy as np
from pydantic import PositiveInt

from einexpr.utils.utils import pedantic_dict_merge
from .exceptions import AmbiguousDimensionException
import einexpr
from lark import Lark, Transformer, v_args

from dataclasses import replace


D = TypeVar("D", bound=einexpr.array_api.dimension.NestedDimension)


def _convert_from_spec(dims):
    if isinstance(dims, einexpr.array_api.dimension.DimensionSpecification):
        return dims.dimensions
    return dims


@einexpr.utils.deprecated_guard
def get_final_aligned_dims(
    *dims: Tuple[einexpr.array_api.dimension.AtomicDimension],
) -> Tuple[einexpr.array_api.dimension.AtomicDimension]:
    """
    Aligns and broadcasts the given dimensions.
    """
    scs = einexpr.utils.get_all_scs_with_nonunique_elems_removed(*dims)
    if len(scs) == 0:
        raise ValueError("No valid alignment found.")
    return next(iter(scs))


@einexpr.utils.deprecated_guard
def calculate_ambiguous_final_aligned_dims(
    *dims: Tuple[einexpr.array_api.dimension.NestedDimension, ...],
) -> Set[einexpr.array_api.dimension.NestedDimension]:
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


@einexpr.utils.deprecated_guard
def get_each_aligned_dims(
    *ein_dims: Tuple[einexpr.array_api.dimension.NestedDimension, ...],
) -> Tuple[Tuple[einexpr.array_api.dimension.Tuple, ...], ...]:
    """
    Returns a list of lists of dimensions that are aligned with eachother.
    """
    aligned_final = get_final_aligned_dims(*ein_dims)
    return tuple(tuple(dim for dim in aligned_final if dim in dims) for dims in ein_dims)


@einexpr.utils.deprecated_guard
def calculate_transexpand(
    dims_from: Tuple[einexpr.array_api.dimension.NestedDimension, ...],
    dims_to: Tuple[einexpr.array_api.dimension.NestedDimension, ...],
) -> Tuple[int | None, ...]:
    """
    Returns lists of the transpositions and expansions required to align to given dimensions. Broadcast dimensions - those
    that will need to be expanded - are represented by None.
    """
    return tuple(dims_from.index(dim) if dim in dims_from else None for dim in dims_to)


@v_args(inline=True)
class TreeToReshapeDimension(Transformer):
    def start(self, root_dims=einexpr.utils.ExpandSentinel()) -> Any:
        return root_dims

    def root_dims(self, dim: D) -> D:
        return dim

    def replacable_dim(self, value) -> einexpr.array_api.dimension.NestedDimension | einexpr.array_api.dimension.DimensionReplacement:
        return value
        
    def dim_replacement(self, original, replacement) -> einexpr.array_api.dimension.DimensionReplacement:
        return einexpr.array_api.dimension.DimensionReplacement(original, replacement)

    def irreplacable_dim(self, value) -> einexpr.array_api.dimension.NestedDimension:
        if isinstance(value, str):
            return value
        elif isinstance(value, tuple):
            return value
        else:
            raise ValueError(f"Unexpected value {value}.")

    def tuple(self, expand_sentinel: einexpr.utils.ExpandSentinel) -> Tuple[einexpr.array_api.dimension.NestedDimension, ...]:
        return tuple(expand_sentinel.content)
    
    def bare_tuple(self, *dims) -> einexpr.utils.ExpandSentinel:
        return einexpr.utils.ExpandSentinel(tuple(dim for dim in dims if dim is not None))
    
    def ellipsis_bare_tuple(self, *dims) -> einexpr.utils.ExpandSentinel:
        return einexpr.utils.ExpandSentinel(tuple(dim for dim in dims if dim is not None))
    
    def sequence(self, *xs) -> Tuple[einexpr.array_api.dimension.NestedDimension, ...]:
        return tuple(x for x in xs if x is not None)

    def sep(self) -> None:
        return None
    
    def named_dim(self, value: str) -> einexpr.array_api.dimension.AtomicDimension:
        return value
    
    def positional_dim(self, value: str) -> einexpr.array_api.dimension.AbsorbingDimension:
        return einexpr.array_api.dimension.AbsorbingDimension()

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

        start: root_dims?
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


@einexpr.utils.deprecated_guard
def parse_dims(dims_raw: einexpr.array_api.dimension.NestedDimension) -> Tuple[einexpr.array_api.dimension.NestedDimension]:
    """
    Parses a dimensions declaration string into a tree.

    Dimensions string example: "you_can_use_commas,or_spaces to_separate_dimensions ( l (m n) ) dimension_identifiers_can_be_any_valid_python_identifier"
    """
    if dims_raw is None:
        return ()

    def helper(dims_raw):
        if isinstance(dims_raw, (tuple, list)):
            children = []
            for child in dims_raw:
                child_processed = helper(child)
                if isinstance(child_processed, einexpr.utils.ExpandSentinel):
                    children.extend(child_processed.content)
                else:
                    children.append(child_processed)
            return tuple(children)
        elif isinstance(dims_raw, einexpr.utils.ExpandSentinel):
            return einexpr.utils.ExpandSentinel(tuple(helper(dim) for dim in dims_raw))
        elif isinstance(dims_raw, einexpr.array_api.dimension.DimensionReplacement):
            return einexpr.array_api.dimension.DimensionReplacement(helper(dims_raw.original), helper(dims_raw.replacement))
        elif isinstance(dims_raw, str):
            tree = dims_reshape_grammar.parse(dims_raw)
            return TreeToReshapeDimension().transform(tree)
        elif isinstance(dims_raw, einexpr.array_api.dimension.AtomicDimension | einexpr.array_api.dimension.AbsorbingDimension | EllipsisType):
            return dims_raw
        elif dims_raw is None:
            return einexpr.array_api.dimension.AbsorbingDimension()
        else:
            raise ValueError(f"Unexpected value {dims_raw}.")
    
    dims = helper(dims_raw)
    if isinstance(dims, einexpr.utils.ExpandSentinel):
        dims = tuple(dims.content)
    elif isinstance(dims_raw, str) or isinstance(dims, einexpr.array_api.dimension.AtomicDimension):
        dims = (dims,)
    dims = helper(dims)
    if not isinstance(dims, tuple):
        raise ValueError(f"Unexpected value {dims}.")
    return dims


@einexpr.utils.deprecated_guard
def parse_dims_reshape(dims_raw: Union[str, Tuple]) -> Tuple:
    """
    Parses a dimensions reshape string into a tree.
    """
    return parse_dims(dims_raw)


@einexpr.utils.deprecated_guard
def dimspec_to_shape(dimspec: einexpr.array_api.dimension.DimensionSpecification) -> Tuple[int, ...]:
    """
    Converts a dimension specification into a shape tuple.
    """
    return dimspec.shape


@einexpr.utils.deprecated_guard
def gather_sizes(dims: einexpr.array_api.dimension.DimensionSpecification) -> Dict[str, int]:
    """
    Returns a dictionary of dimension names and sizes.
    """
    if not isinstance(dims, einexpr.array_api.dimension.DimensionSpecification):
        raise TypeError(f"Expected a DimensionSpecification, got {dims}.")
    return dims.sizes


@einexpr.utils.deprecated_guard
def process_dims_declaration(dims_raw: str | Tuple[einexpr.array_api.dimension.BaseDimension, ...], shape: Tuple[int, ...]) -> einexpr.array_api.dimension.DimensionSpecification:
    """
    Processes a dimensions declaration into a dimension object.
    """
    if dims_raw is None:
        # Use full tuple of absorbing dimensions
        dims_raw = tuple(einexpr.array_api.dimension.AbsorbingDimension() for _ in shape)
    elif not isinstance(dims_raw, (str, tuple, list)):
        raise ValueError(f"Unexpected value {dims_raw}.")
    dims = parse_dims(dims_raw)
    # Expand ellipsis
    if ... in dims:
        i_ellipsis = dims.index(...)
        len_ellipsis = len(shape) - (len(dims) - 1)
        dims = dims[:i_ellipsis] + tuple(einexpr.array_api.dimension.AbsorbingDimension() for _ in range(len_ellipsis)) + dims[i_ellipsis + 1:]
    if len(dims) != len(shape):
        raise einexpr.exceptions.InternalError("DimensionSpecification tuple size does not match shape tuple size.")
    if len(dims) != len(shape):
        raise ValueError(f"Declaration {dims_raw} has wrong number of dimensions. Expected {len(shape)}, got {len(dims)}.")
    dim_sizes = {dim: size for dim, size in zip(dims, shape)}
    dims = einexpr.array_api.dimension.DimensionSpecification(dims, sizes=dim_sizes)
    return dims


@einexpr.utils.deprecated_guard
def ignore_replacements(dims: D) -> D:
    """
    Replaces all dimension replacements with their original dimensions.
    """
    if isinstance(dims, einexpr.array_api.dimension.DimensionSpecification):
        return einexpr.array_api.dimension.DimensionSpecification(ignore_replacements(dims.dimensions), sizes=dims.sizes)
    elif isinstance(dims, einexpr.array_api.dimension.DimensionReplacement):
        return ignore_replacements(dims.original)
    elif isinstance(dims, tuple):
        return tuple(ignore_replacements(dim) for dim in dims)
    elif isinstance(dims, einexpr.array_api.dimension.AtomicDimension | einexpr.array_api.dimension.AbsorbingDimension | EllipsisType):
        return dims
    else:
        raise ValueError(f"Unexpected value {dims}.")


@einexpr.utils.deprecated_guard
def apply_replacements(dims: D) -> D:
    """
    Replaces all dimension replacements with their replacement dimensions.
    """
    if isinstance(dims, einexpr.array_api.dimension.DimensionReplacement):
        return apply_replacements(dims.replacement)
    elif isinstance(dims, tuple):
        return tuple(apply_replacements(dim) for dim in dims)
    elif isinstance(dims, einexpr.array_api.dimension.AtomicDimension | einexpr.array_api.dimension.AbsorbingDimension | EllipsisType):
        return dims
    elif isinstance(dims, einexpr.array_api.dimension.DimensionSpecification):
        return einexpr.array_api.dimension.DimensionSpecification(apply_replacements(dims.dimensions), sizes=dims.sizes)
    else:
        raise ValueError(f"Unexpected value {dims!r}.")


DimTupleOrSpec = TypeVar('DimTupleOrSpec', bound=Tuple[einexpr.array_api.dimension.NestedDimension, ...] | einexpr.array_api.dimension.DimensionSpecification)

@einexpr.utils.deprecated_guard
def expand_dims(dims: DimTupleOrSpec) -> DimTupleOrSpec:
    """
    Expands all composite dimensions into their constituents, effectually 'flattening' the dimension tree itself.
    """
    def helper(dims: D) -> D:
        if isinstance(dims, einexpr.array_api.dimension.DimensionReplacement):
            raise ValueError(f"Can't expand a dimension replacement: {dims}. Use ignore_replacements() or apply_replacements() first.")
        elif isinstance(dims, tuple):
            for dim in dims:
                yield from helper(dim)
        elif isinstance(dims, einexpr.array_api.dimension.AtomicDimension | einexpr.array_api.dimension.AbsorbingDimension):
            yield dims
        else:
            raise ValueError(f"Unexpected value {dims}.")
    if isinstance(dims, einexpr.array_api.dimension.DimensionSpecification):
        return einexpr.array_api.dimension.DimensionSpecification(helper(dims.dimensions), dims.sizes)
    else:
        return tuple(helper(dims))


@einexpr.utils.deprecated_guard
def isolate_dim(array_dims: Tuple[einexpr.array_api.BaseDimension, ...], dim: einexpr.array_api.dimension.BaseDimension, split_mode: Literal['left', 'middle', 'right']) -> Tuple[einexpr.array_api.BaseDimension, ...]:
    """
    Extract the dimensions in ``dim`` into the first level if it is part of a composite dimension.
    """
    position = einexpr.utils.get_position(array_dims, dim)
    if position is None:
        raise ValueError(f"Dimension {dim!r} is not part of the array dimensions {array_dims!r}.")
    if split_mode == 'middle':
        lhs, rhs = einexpr.utils.split_at(array_dims, position)
        return einexpr.utils.deep_remove((*lhs, dim, *rhs), lambda x: x == ())
    elif split_mode == 'left':
        return einexpr.utils.deep_remove((dim, *einexpr.utils.deep_remove(array_dims, lambda x: x == dim)), lambda x: x == ())
    elif split_mode == 'right':
        return einexpr.utils.deep_remove((*einexpr.utils.deep_remove(array_dims, dim), dim), lambda x: x == ())


@einexpr.utils.deprecated_guard
def isolate_dims(array_dims: Tuple[einexpr.array_api.BaseDimension, ...], dims: Tuple[einexpr.array_api.dimension.BaseDimension, ...], split_mode: Literal['left', 'middle', 'right'] = 'middle') -> Tuple[einexpr.array_api.BaseDimension, ...]:
    """
    Extract the dimensions in ``dims`` into the first level if they are part of a composite dimension.
    """
    for dim in dims:
        array_dims = isolate_dim(array_dims, dim, split_mode)
    return array_dims


def is_dimensionless(array):
    return isinstance(array, (int, float)) or einexpr.backends.conforms_to_array_api(array) and len(array.shape) == 0


@einexpr.utils.deprecated_guard
def get_dimspec(array):
    if isinstance(array, einexpr.array):
        return array.dimspec
    elif is_dimensionless(array):
        return einexpr.array_api.dimension.DimensionSpecification((), {})
    else:
        raise TypeError(f'{array} is not a recognized einexpr array')


@overload
def resolve_positional_dims(dimensions_tuples: Tuple[einexpr.array_api.dimension.DimensionSpecification, ...]) -> Tuple[einexpr.array_api.dimension.DimensionSpecification, ...]:
    ...


@overload
def resolve_positional_dims(dimensions_tuples: Tuple[Tuple[einexpr.array_api.dimension.BaseDimension, ...], ...]) -> Tuple[Tuple[einexpr.array_api.dimension.BaseDimension, ...], ...]:
    ...


@einexpr.utils.deprecated_guard
def resolve_positional_dims(dimensions_tuples):
    """
    For each absorbing dimension in each dimension specification, absorb a corresponding non-absorbable dimension from another dimension specification. Dimensions are resolved in broadcast order (i.e. right-to-left).
    """
    if all(isinstance(dimensions, einexpr.array_api.dimension.DimensionSpecification) for dimensions in dimensions_tuples):
        _dimensions_tuples = resolve_positional_dims(tuple(dimspec.dimensions for dimspec in dimensions_tuples))
        return tuple(dimspec.with_dimensions(dimensions) for dimspec, dimensions in zip(dimensions_tuples, _dimensions_tuples))
    elif not any(isinstance(dimensions, einexpr.array_api.dimension.DimensionSpecification) for dimensions in dimensions_tuples):
        # TODO: This function has a huge amount of redundant code.
        # Replace absorbing dimensions by `None`
        dimensions_tuples = tuple(tuple(None if isinstance(dim, einexpr.array_api.dimension.AbsorbingDimension) else dim for dim in dimensions) for dimensions in dimensions_tuples)
        _dimensions_tuples = []
        for _absorbing_dimensions in dimensions_tuples:
            absorbing_dimensions = list(_absorbing_dimensions)
            for dimensions_to_absorb in dimensions_tuples:
                if _absorbing_dimensions is dimensions_to_absorb:
                    continue
                if ... not in absorbing_dimensions and ... not in dimensions_to_absorb:
                    if len(absorbing_dimensions) != len(dimensions_to_absorb):
                        raise ValueError(f"Dimension tuples without ellipses must have the same lengths: {absorbing_dimensions!r} != {dimensions_to_absorb!r}.")
                    absorbing_dimensions = tuple(dimension_to_absorb if absorbing_dimension is None else absorbing_dimension for absorbing_dimension, dimension_to_absorb in zip(absorbing_dimensions, dimensions_to_absorb))
                else:
                    dimensions_to_absorb = tuple(dimension_to_absorb for dimension_to_absorb in dimensions_to_absorb if dimension_to_absorb is not None)
                    common_dimensions = set(absorbing_dimensions) & set(dimensions_to_absorb) - {..., None}
                    dimensions_to_absorb = tuple(dimension_to_absorb for dimension_to_absorb in dimensions_to_absorb if dimension_to_absorb not in common_dimensions)
                    if ... in absorbing_dimensions and ... not in dimensions_to_absorb:
                        i_absorbing_dimensions_ellipsis = absorbing_dimensions.index(...)
                        absorbing_dimensions_lhs = absorbing_dimensions[:i_absorbing_dimensions_ellipsis]
                        absorbing_dimensions_rhs = absorbing_dimensions[i_absorbing_dimensions_ellipsis + 1:]
                        dimensions_to_absorb_lhs = iter(dimensions_to_absorb)
                        dimensions_to_absorb_rhs = iter(reversed(dimensions_to_absorb))
                        for i, absorbing_dimension in enumerate(absorbing_dimensions_lhs):
                            if absorbing_dimension is None:
                                try:
                                    dimension_to_absorb = next(dimensions_to_absorb_rhs)
                                except StopIteration:
                                    # TODO: this should never be triggered here due to dimensions length requirements.
                                    break
                                else:
                                    absorbing_dimensions[-i] = dimension_to_absorb
                        for i, absorbing_dimension in enumerate(reversed(absorbing_dimensions_rhs), start=1):
                            if absorbing_dimension is None:
                                try:
                                    dimension_to_absorb = next(dimensions_to_absorb_rhs)
                                except StopIteration:
                                    break
                                else:
                                    absorbing_dimensions[-i] = dimension_to_absorb
                    elif ... not in absorbing_dimensions and ... in dimensions_to_absorb:
                        i_dimensions_to_absorb_ellipsis = dimensions_to_absorb.index(...)
                        absorbing_dimensions_lhs = absorbing_dimensions
                        absorbing_dimensions_rhs = absorbing_dimensions
                        dimensions_to_absorb_lhs = iter(dimensions_to_absorb[:i_dimensions_to_absorb_ellipsis])
                        dimensions_to_absorb_rhs = iter(reversed(dimensions_to_absorb[i_dimensions_to_absorb_ellipsis + 1:]))
                        for i, absorbing_dimension in enumerate(absorbing_dimensions_lhs):
                            if absorbing_dimension is None:
                                try:
                                    dimension_to_absorb = next(dimensions_to_absorb_lhs)
                                except StopIteration:
                                    break
                                else:
                                    absorbing_dimensions[i] = dimension_to_absorb
                        for i, absorbing_dimension in enumerate(reversed(absorbing_dimensions_rhs), start=1):
                            if absorbing_dimension is None:
                                try:
                                    dimension_to_absorb = next(dimensions_to_absorb_rhs)
                                except StopIteration:
                                    break
                                else:
                                    absorbing_dimensions[-i] = dimension_to_absorb
                    elif ... in absorbing_dimensions and ... in dimensions_to_absorb:
                        i_absorbing_dimensions_ellipsis = absorbing_dimensions.index(...)
                        absorbing_dimensions_lhs = absorbing_dimensions[:i_absorbing_dimensions_ellipsis]
                        absorbing_dimensions_rhs = absorbing_dimensions[i_absorbing_dimensions_ellipsis + 1:]
                        i_dimensions_to_absorb_ellipsis = dimensions_to_absorb.index(...)
                        dimensions_to_absorb_lhs = iter(dimensions_to_absorb[:i_dimensions_to_absorb_ellipsis])
                        dimensions_to_absorb_rhs = iter(reversed(dimensions_to_absorb[i_dimensions_to_absorb_ellipsis + 1:]))
                        for i, absorbing_dimension in enumerate(absorbing_dimensions_lhs):
                            if absorbing_dimension is None:
                                try:
                                    dimension_to_absorb = next(dimensions_to_absorb_lhs)
                                except StopIteration:
                                    break
                                else:
                                    absorbing_dimensions[i] = dimension_to_absorb
                        for i, absorbing_dimension in enumerate(reversed(absorbing_dimensions_rhs), start=1):
                            if absorbing_dimension is None:
                                try:
                                    dimension_to_absorb = next(dimensions_to_absorb_rhs)
                                except StopIteration:
                                    break
                                else:
                                    absorbing_dimensions[-i] = dimension_to_absorb
                    else:
                        raise einexpr.exceptions.InternalError(f"Should never get here.")
            _dimensions_tuples.append(absorbing_dimensions)
        dimensions_tuples = _dimensions_tuples
        len_dimensions = None
        for dimensions in dimensions_tuples:
            if ... not in dimensions:
                len_dimensions = len(dimensions)
                break
        _dimensions_tuples = []

        if len_dimensions is None:
            absorbable_dimension_generator_lhs = (einexpr.array_api.dimension.AbsorbingDimension() for _ in itertools.repeat(None))
            absorbable_dimension_generator_rhs = (einexpr.array_api.dimension.AbsorbingDimension() for _ in itertools.repeat(None))
            for dimensions in dimensions_tuples:
                i_ellipsis = dimensions.index(...)
                absorbable_dimension_generator_lhs, absorbable_dimension_generator_instance_lhs = itertools.tee(absorbable_dimension_generator_lhs)
                absorbable_dimension_generator_rhs, absorbable_dimension_generator_instance_rhs = itertools.tee(absorbable_dimension_generator_rhs)
                dimensions_lhs = tuple(next(absorbable_dimension_generator_instance_lhs) if dimension is None else dimension for dimension in dimensions[:i_ellipsis])
                dimensions_rhs = reversed(tuple(next(absorbable_dimension_generator_instance_rhs) if dimension is None else dimension for dimension in reversed(dimensions[i_ellipsis + 1:])))
                _dimensions_tuples.append((*dimensions_lhs, ..., *dimensions_rhs))
        else:
            # TODO: Ensure the lengths of all ellipsis-containing dimensions are less than or equal to the length of the longest dimension.
            #       Might need to do this earlier.
            num_absorbables = next(dimensions.count(None) for dimensions in dimensions_tuples if ... not in dimensions)
            for dimensions in dimensions_tuples:
                if ... in dimensions:
                    if dimensions.count(None) > num_absorbables:
                        raise einexpr.exceptions.InternalError(f"All dimension tuples that don't contain an ellipsis must have the same number of absorbing dimensions, and this number must be greater than the number of absorbing dimensions in any of the ellipsis-containing dimension tuples. Got {dimensions.count(None)} absorbing dimensions in {dimensions}; expected less than {num_absorbables}.")
                elif dimensions.count(None) != num_absorbables:
                    raise einexpr.exceptions.InternalError(f"All dimension tuples that don't contain an ellipsis must have the same number of absorbing dimensions, and this number must be greater than the number of absorbing dimensions in any of the ellipsis-containing dimension tuples. Got {dimensions.count(None)} absorbing dimensions in {dimensions}; expected {num_absorbables}.")
            absorbables = [einexpr.array_api.dimension.AbsorbingDimension() for _ in range(num_absorbables)]
            for dimensions in dimensions_tuples:
                if ... in dimensions:
                    i_ellipsis = dimensions.index(...)
                    absorbable_dimension_generator_lhs = iter(absorbables)
                    absorbable_dimension_generator_rhs = iter(reversed(absorbables))
                    dimensions_lhs = tuple(next(absorbable_dimension_generator_lhs) if dimension is None else dimension for dimension in dimensions[:i_ellipsis])
                    dimensions_rhs = tuple(next(absorbable_dimension_generator_rhs) if dimension is None else dimension for dimension in dimensions[i_ellipsis + 1:])
                    dimensions = ((*dimensions_lhs, ..., *dimensions_rhs))
                else:
                    absorbable_dimension_generator = iter(absorbables)
                    dimensions = tuple(next(absorbable_dimension_generator) if dimension is None else dimension for dimension in dimensions)
                _dimensions_tuples.append(dimensions)
        return tuple(_dimensions_tuples)
    else:
        raise ValueError(f"Cannot resolve positional dimensions for {dimensions_tuples}.")