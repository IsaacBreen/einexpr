import functools
import operator
import re
from itertools import chain, zip_longest
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
from pydantic import PositiveInt
from einexpr.array_api.dimension import NamedBindingDimension

from einexpr.utils.utils import pedantic_dict_merge
from .exceptions import AmbiguousDimensionException
import einexpr
from lark import Lark, Transformer, v_args


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
class TreeToDimsDeclaration(Transformer):
    def start(self, x) -> Any:
        return x

    def dims(self, *dims) -> einexpr.array_api.dimension.DimensionTuple:
        return einexpr.array_api.dimension.DimensionTuple(dim for dim in dims if dim is not None)

    def dim(self, value, *x) -> Union[einexpr.array_api.dimension.Dimension, Tuple]:
        return value

    def tuple(self, *dims) -> einexpr.array_api.dimension.DimensionTuple:
        return einexpr.array_api.dimension.DimensionTuple(dim for dim in dims if dim is not None)

    def sep(self) -> None:
        return None

    def NAME(self, tree) -> einexpr.array_api.dimension.Dimension:
        return tree.value



dims_declaration_grammar = Lark(
    r"""
        %import common.WS

        %ignore WS

        start: dims
        ?dims: (dim [sep])*
        ?dim: NAME | "(" dim ")" | tuple{dim}
        tuple{x}: "(" x (sep | ([sep] x)+ [sep]) ")"
        ?sep: ","
        NAME: /[^\W\d]\w*/
    """,
    parser="earley",
    maybe_placeholders=False,
)

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
    
    def dim(self, value) -> einexpr.array_api.dimension.NamedBindingDimension:
        return einexpr.array_api.dimension.NamedBindingDimension(value)

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


def parse_dims_declaration(dims_raw: Union[str, Tuple]) -> Tuple:
    """
    Parses a dimensions declaration string into a tree.

    Dimensions string example: "you_can_use_commas,or_spaces to_separate_dimensions ( l (m n) ) dimension_identifiers_can_be_any_valid_python_identifier"
    """
    if not dims_raw:
        return ()

    def helper(dims_raw):
        if isinstance(dims_raw, (tuple, list)):
            return einexpr.array_api.dimension.DimensionTuple((helper(dim) for dim in dims_raw), size=compute_total_size(dims_raw))
        else:
            tree = dims_declaration_grammar.parse(dims_raw)
            return TreeToDimsDeclaration().transform(tree)
    dims = helper(dims_raw)
    if isinstance(dims, einexpr.array_api.dimension.Dimension):
        return (dims,)
    else:
        return dims


def parse_dims_reshape(dims_raw: Union[str, Tuple]) -> Tuple:
    """
    Parses a dimensions reshape string into a tree.
    """
    if not dims_raw:
        return ()

    def helper(dims_raw):
        if isinstance(dims_raw, (tuple, list)):
            return einexpr.array_api.dimension.DimensionTuple((helper(dim) for dim in dims_raw), size=compute_total_size(dims_raw))
        else:
            tree = dims_reshape_grammar.parse(dims_raw)
            return TreeToReshapeDimension().transform(tree)

    dims = helper(dims_raw)
    if isinstance(dims, einexpr.array_api.dimension.Dimension):
        return (dims,)
    else:
        return dims


def gather_dims(dims: einexpr.array_api.dimension.Dimensions) -> Set[einexpr.array_api.dimension.Dimension]:
    """
    Returns all individual dimensions in the object, including nested ones.
    """
    if isinstance(dims, einexpr.array_api.dimension.Dimension):
        return {dims}
    elif isinstance(dims, einexpr.array_api.dimension.DimensionReplacement):
        return gather_dims(dims.original) | gather_dims(dims.replacement)
    elif isinstance(dims, tuple):
        return set(chain.from_iterable(gather_dims(dim) for dim in dims))
    else:
        raise ValueError(f"Unexpected value {dims}.")


def compute_total_size(dims: einexpr.array_api.dimension.Dimensions) -> Optional[int]:
    """
    Returns the total dimensions size, or None if the dimensions are not known.
    """
    if isinstance(dims, einexpr.array_api.dimension.Dimension):
        return dims.size
    elif isinstance(dims, einexpr.array_api.dimension.DimensionReplacement):
        return compute_total_size(dims.replacement)
    elif isinstance(dims, (tuple, einexpr.array_api.dimension.DimensionTuple)):
        size = 1
        for dim in dims:
            if dim.size is None:
                return None
            size *= dim.size
        return size
    else:
        raise ValueError(f"Unexpected value {dims}.")


def infer_sizes(dims: einexpr.array_api.dimension.DimensionTuple, overall_size: PositiveInt) -> Tuple[int, ...]:
    """
    If there is exactly one dimension with an undefined size, infer that size and return it and the rest of the sizes.
    """
    # If there is not exactly one dimension with an undefined size, we cannot or do not need to infer anything.
    if dims.count(None) != 1:
        return dims
    # Get the index of the dimension with an undefined size.
    sizes = [dim.size for dim in dims]
    undefined_dim_index = sizes.index(None)
    # Set the size of this dimension to 1 and calculate the product of the other dimensions.
    sizes[undefined_dim_index] = 1
    product = functools.reduce(operator.mul, sizes, 1)
    # The overall size must be evenly divisible by the product of the other dimensions.
    if overall_size % product != 0:
        raise ValueError(f"The overall size {overall_size} is not evenly divisible by the product of the other dimensions {product}.")
    # Infer the size of the undefined dimension.
    sizes[undefined_dim_index] = overall_size // product
    return tuple(sizes)


def iter_dims(dims: einexpr.array_api.dimension.DimensionObject) -> Iterator[einexpr.array_api.dimension.Dimension]:
    """
    Iterates over all dimension objects.
    """
    yield dims
    if isinstance(dims, einexpr.array_api.dimension.DimensionReplacement):
        yield from iter_dims(dims.original)
        yield from iter_dims(dims.replacement)
    elif isinstance(dims, (tuple, einexpr.array_api.dimension.DimensionTuple)):
        for dim in dims:
            yield from iter_dims(dim)
    else:
        raise ValueError(f"Unexpected value {dims}.")


def gather_sized_dims(dims: einexpr.array_api.dimension.DimensionObject) -> Set[einexpr.array_api.dimension.DimensionObject]:
    """
    Returns the set of all dimension objects that have a defined size. Ignores dimensions that don't have a defined size and raises an error if two dimensions with the same have inconsistent sizes.
    """
    sizes = set()
    named_sizes = {}
    for dim_obj in iter_dims(dims):
        if isinstance(dim_obj, einexpr.array_api.dimension.Sized):
            if isinstance(dim_obj, einexpr.array_api.dimension.NamedBindingDimension):
                if dim_obj.name in named_sizes:
                    if named_sizes[dim_obj.name] != dim_obj.size:
                        raise ValueError(f"The size of dimension {dim_obj.name} is inconsistent with the previously encountered size {named_sizes[dim_obj.name]}.")
                else:
                    named_sizes[dim_obj.name] = dim_obj.size
            sizes.add(dim_obj)
    return sizes


def simplify_dims(dims: einexpr.array_api.dimension.Dimensions, dim_sizes: Optional[Dict[str, int]] = None) -> einexpr.array_api.dimension.Dimensions:
    """
    Adds sizes to dimensions that are missing them.
    """
    dim_sizes = dim_sizes or {}
    dim_sizes = {dim.name if isinstance(dim, einexpr.array_api.dimension.NamedBindingDimension) else dim: size for dim, size in dim_sizes.items()}
    existing_sizes = {dim.name if isinstance(dim, einexpr.array_api.dimension.NamedBindingDimension) else dim: dim.size for dim in gather_sized_dims(dims)}
    dim_sizes = einexpr.utils.pedantic_dict_merge(dim_sizes, existing_sizes)
    def helper(dims):
        if isinstance(dims, einexpr.array_api.dimension.NamedBindingDimension):
            if dims.name in dim_sizes:
                if dims.size is not None and dims.size != dim_sizes[dims.name]:
                    # TODO: This should never get triggered.
                    raise ValueError(f"Expected dimension {dims} to have size {dim_sizes[dims]}, but got {dims.size}.")
                dims.size = dim_sizes[dims.name]
            else:
                return dims
        elif isinstance(dims, einexpr.array_api.dimension.DimensionTuple):
            if any(isinstance(dim, einexpr.array_api.dimension.Sized) and dim.size is None for dim in dims) and dims.size is not None:
                dims = dims.with_size(dims.size)
        elif isinstance(dims, einexpr.array_api.dimension.DimensionReplacement):
            # NOTE: Even though we don't, strictly speaking, need the passed dim_sizes to main consistency with the dims.original dims,
            #       it's still a good idea to call simplify_dims on them just to help alert the user to any inconsistencies that may
            #       cause problems at other points of their code.
            original_dims = helper(dims.original)
            replacement_dims = helper(dims.replacement)
        
        
            
        elif isinstance(dims, (tuple, einexpr.array_api.dimension.DimensionTuple)):
            return einexpr.array_api.dimension.DimensionTuple((simplify_dims(dim, dim_sizes) for dim in dims), size=compute_total_size(dims))
        else:
            raise ValueError(f"Unexpected value {dims}.")