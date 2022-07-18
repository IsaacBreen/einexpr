import re
from itertools import zip_longest
from typing import Any, Dict, List, Sequence, Set, Tuple, Union

import numpy as np
from .exceptions import AmbiguousDimensionException
import einexpr
from lark import Lark, Transformer, v_args


def get_unambiguous_broadcast_dims(*dims: List[einexpr.types.Dimension]) -> Set[einexpr.types.Dimension]:
    scs = iter(einexpr.get_all_scs_with_unique_elems(*dims))
    first_broadcasted_dims = np.array(list(next(scs)))
    unambiguous_positions = np.ones((len(first_broadcasted_dims),), dtype=bool)
    for broadcasted_dims in scs:
        unambiguous_positions &= first_broadcasted_dims == np.array(list(broadcasted_dims))
    return set(first_broadcasted_dims[unambiguous_positions].tolist())


def get_ambiguous_broadcast_dims(*dims: List[einexpr.types.Dimension]) -> Set[einexpr.types.Dimension]:
    scs = iter(einexpr.get_all_scs_with_unique_elems(*dims))
    first_broadcasted_dims = np.array(list(next(scs)))
    ambiguous_positions = np.zeros((len(first_broadcasted_dims),), dtype=bool)
    for broadcasted_dims in scs:
        ambiguous_positions |= first_broadcasted_dims != np.array(list(broadcasted_dims))
    return set(first_broadcasted_dims[ambiguous_positions].tolist())


def get_unique_broadcast(*dims: List[einexpr.types.Dimension]) -> Set[einexpr.types.Dimension]:
    scs = einexpr.get_all_scs_with_unique_elems(*dims)
    if len(scs) == 0:
        raise ValueError("No valid broadcast found.")
    if len(scs) > 1:
        raise AmbiguousDimensionException("Multiple valid broadcasts found.")
    return scs.pop()


def get_any_broadcast(*dims: List[einexpr.types.Dimension]) -> Set[einexpr.types.Dimension]:
    """
    Use this when you need a broadcast but don't care about its uniqueness. The broadcast chosen is deterministic and depends only on the order dimensions within each argument and the order in which the arguments are passed.
    """
    scs = einexpr.get_all_scs_with_unique_elems(*dims)
    if len(scs) == 0:
        raise ValueError("No valid broadcast found.")
    return sorted(scs).pop()


def get_final_aligned_dims(*ein_dims: List[einexpr.types.Dimension]) -> List[einexpr.types.Dimension]:
    """
    Aligns and broadcasts the given dimensions.
    """
    scs = einexpr.utils.get_all_scs_with_nonunique_elems_removed(*ein_dims)
    if len(scs) == 0:
        raise ValueError("No valid alignment found.")
    return sorted(scs).pop()


def calculate_ambiguous_final_aligned_dims(*dims: List[einexpr.types.Dimension]) -> Set[einexpr.types.Dimension]:
    aligned_dims = get_each_aligned_dims(*dims)
    scs = list(einexpr.get_all_scs_with_unique_elems(*aligned_dims))
    if len(scs) == 0:
        raise ValueError(f"No valid alignments for the given dimensions {dims}.")
    first_broadcasted_dims = np.array(scs[0])
    ambiguous_positions = np.zeros((len(first_broadcasted_dims),), dtype=bool)
    for broadcasted_dims in scs[1:]:
        ambiguous_positions |= first_broadcasted_dims != np.array(list(broadcasted_dims))
    return set(first_broadcasted_dims[ambiguous_positions].tolist())


def get_each_aligned_dims(*ein_dims: List[einexpr.types.Dimension]) -> List[List[einexpr.types.Dimension]]:
    """
    Returns a list of lists of dimensions that are aligned with eachother.
    """
    aligned_final = get_final_aligned_dims(*ein_dims)
    return [[dim for dim in aligned_final if dim in dims] for dims in ein_dims]


def calculate_transexpand(dims_from: List[einexpr.types.Dimension], dims_to: List[einexpr.types.Dimension]) -> List[einexpr.types.Dimension]:
    """
    Returns lists of the transpositions and expansions required to align to given dimensions. Broadcast dimensions - those
    that will need to be expanded - are represented by None.
    """
    return [dims_from.index(dim) if dim in dims_from else None for dim in dims_to]


@v_args(inline=True)
class TreeToDimsDeclaration(Transformer):
    def dims(self, *dims) -> Tuple[einexpr.types.Dimension]:
        return tuple(dim for dim in dims if dim is not None)
    
    def dim(self, value, *x) -> Union[einexpr.types.Dimension, Tuple]:
        print(x)
        return value
    
    def tuple(self, *dims) -> Tuple[einexpr.types.Dimension]:
        return tuple(dim for dim in dims if dim is not None)
    
    def sep(self) -> None:
        return None

    def NAME(self, tree) -> einexpr.types.Dimension:
        return tree.value
    

dims_declaration_grammar = Lark(r'''
                    %import common.WS
                    
                    %ignore WS
                    
                    ?start: dims
                    ?dims: (dim [sep])*
                    ?dim: tuple | NAME | "(" dim ")"
                    tuple: "(" dim (sep | ([sep] dim)+ [sep]) ")"
                    ?sep: ","
                    NAME: /[^\W\d]\w*/
                    ''',
                    parser='lalr',
                    transformer=TreeToDimsDeclaration(),
                    maybe_placeholders=False)


@v_args(inline=True)
class TreeToReshape(Transformer):
    def dims(self, *dims) -> Tuple[einexpr.types.Dimension]:
        return tuple(dim for dim in dims if dim is not None)
    
    def dim(self, value, *x) -> Union[einexpr.types.Dimension, Tuple]:
        print(x)
        return value
    
    def tuple(self, *dims) -> Tuple[einexpr.types.Dimension]:
        return tuple(dim for dim in dims if dim is not None)
    
    def sep(self) -> None:
        return None

    def NAME(self, tree) -> einexpr.types.Dimension:
        return tree.value


dims_reshape_grammar = Lark(r'''
                    %import common.WS
                    
                    %ignore WS
                    
                    ?start: dims
                    ?dims: (dim [sep])*
                    ?dim: ( tuple | NAME | "(" dim ")" ) ( "->" dim )?
                    tuple: "(" dim (sep | ([sep] dim)+ [sep]) ")"
                    ?sep: ","
                    NAME: /[^\W\d]\w*/
                    ''',
                    parser='lalr',
                    transformer=TreeToReshape(),
                    maybe_placeholders=False)


dims_reshape_grammar.parse('j i j,f,(fdsf,) x->j (k)->f q->(i f2 (x y( z)))')


def parse_dims_declaration(dims_raw: Union[str, Tuple]) -> Tuple:
    """
    Parses a dimensions declaration string into a tree. 
    
    Dimensions string example: "you_can_use_commas,or_spaces to_separate_dimensions ( l (m n) ) dimension_identifiers_can_be_any_valid_python_identifier"
    """
    if not dims_raw:
        return ()
    def helper(dims_raw):
        if isinstance(dims_raw, (tuple, list)):
            return tuple(helper(dim) for dim in dims_raw)
        else:
            return dims_declaration_grammar.parse(dims_raw)
    dims = helper(dims_raw)
    if isinstance(dims, einexpr.types.Dimension):
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
            return tuple(helper(dim) for dim in dims_raw)
        else:
            return dims_reshape_grammar.parse(dims_raw)
    dims = helper(dims_raw)
    if isinstance(dims, einexpr.types.Dimension):
        return (dims,)
    else:
        return dims
