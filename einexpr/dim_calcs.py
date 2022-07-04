import re
from itertools import zip_longest
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np
from lark import Lark, Transformer, v_args

from einexpr.base_typing import Dimension
from einexpr.exceptions import AmbiguousDimensionException

from .parse_numpy_ufunc_signature import (UfuncSignature,
                                          UfuncSignatureDimensions,
                                          parse_ufunc_signature)
from .typing import ConcreteArrayLike, RawArrayLike
from .utils import *
from .utils import get_all_scs_with_unique_elems, powerset


def get_unambiguous_broadcast_dims(*dims: List[Dimension]) -> Set[Dimension]:
    scs = iter(get_all_scs_with_unique_elems(*dims))
    first_broadcasted_dims = np.array(list(next(scs)))
    unambiguous_positions = np.ones((len(first_broadcasted_dims),), dtype=bool)
    for broadcasted_dims in scs:
        unambiguous_positions &= first_broadcasted_dims == np.array(list(broadcasted_dims))
    return set(first_broadcasted_dims[unambiguous_positions].tolist())


def get_ambiguous_broadcast_dims(*dims: List[Dimension]) -> Set[Dimension]:
    scs = iter(get_all_scs_with_unique_elems(*dims))
    first_broadcasted_dims = np.array(list(next(scs)))
    ambiguous_positions = np.zeros((len(first_broadcasted_dims),), dtype=bool)
    for broadcasted_dims in scs:
        ambiguous_positions |= first_broadcasted_dims != np.array(list(broadcasted_dims))
    return set(first_broadcasted_dims[ambiguous_positions].tolist())


def get_unique_broadcast(*dims: List[Dimension]) -> Set[Dimension]:
    scs = get_all_scs_with_unique_elems(*dims)
    if len(scs) == 0:
        raise ValueError("No valid broadcast found.")
    if len(scs) > 1:
        raise AmbiguousDimensionException("Multiple valid broadcasts found.")
    return scs.pop()


def get_any_broadcast(*dims: List[Dimension]) -> Set[Dimension]:
    """
    Use this when you need a broadcast but don't care about its uniqueness. The broadcast chosen is deterministic and depends only on the order dimensions within each argument and the order in which the arguments are passed.
    """
    scs = get_all_scs_with_unique_elems(*dims)
    if len(scs) == 0:
        raise ValueError("No valid broadcast found.")
    return sorted(scs).pop()


def get_final_aligned_dims(*ein_dims: List[Dimension]) -> List[Dimension]:
    """
    Aligns and broadcasts the given dimensions.
    """
    scs = get_all_scs_with_nonunique_elems_removed(*ein_dims)
    if len(scs) == 0:
        raise ValueError("No valid alignment found.")
    return sorted(scs).pop()


def calculate_ambiguous_final_aligned_dims(*dims: List[Dimension]) -> Set[Dimension]:
    aligned_dims = get_each_aligned_dims(*dims)
    scs = list(get_all_scs_with_unique_elems(*aligned_dims))
    if len(scs) == 0:
        raise ValueError(f"No valid alignments for the given dimensions {dims}.")
    first_broadcasted_dims = np.array(scs[0])
    ambiguous_positions = np.zeros((len(first_broadcasted_dims),), dtype=bool)
    for broadcasted_dims in scs[1:]:
        ambiguous_positions |= first_broadcasted_dims != np.array(list(broadcasted_dims))
    return set(first_broadcasted_dims[ambiguous_positions].tolist())


def get_each_aligned_dims(*ein_dims: List[Dimension]) -> List[List[Dimension]]:
    """
    Returns a list of lists of dimensions that are aligned with eachother.
    """
    aligned_final = get_final_aligned_dims(*ein_dims)
    return [[dim for dim in aligned_final if dim in dims] for dims in ein_dims]

def dims_are_aligned(dims1: List[Dimension], dims2: List[Dimension]) -> bool:
    """
    Returns True if the given dimensions are aligned.
    """
    return [dim for dim in dims1 if dim in dims2] == [dim for dim in dims2 if dim in dims1]


def calculate_transexpand(dims_from: List[Dimension], dims_to: List[Dimension]) -> List[Dimension]:
    """
    Returns lists of the transpositions and expansions required to align to given dimensions. Broadcast dimensions - those
    that will need to be expanded - are represented by None.
    """
    return [dims_from.index(dim) if dim in dims_from else None for dim in dims_to]


def compute_active_signature_optionals(signature: UfuncSignature, *input_dims: List[Dimension]) -> Set[str]:
    # TODO: This is probably not actually how numpy chooses the active optionals. It very likely resolves the ambiguity of which set of optionals to use (there can be more than one correct set) differently.
    if len(input_dims) != len(signature.input_dims):
        raise ValueError(f"The number {len(input_dims)} of input dimensions passed as arguments ({input_dims}) does not match the number ({len(signature.input_dims)}) of input dimensions specified in the signature {signature}.")
    optional_dim_names = {dim.name for dims in (signature.output_dims, *signature.input_dims) for dim in dims.dims if dim.optional}
    for active_optionals in powerset(optional_dim_names, reverse=True):
        break_flag = False # break on failure
        dim_map = {}
        for inp, ufunc_inp_sigs in zip_longest(input_dims, signature.input_dims):
            inp_dims = list(inp)
            sig_dims = [dim for dim in ufunc_inp_sigs.dims if not (dim.optional and dim.name not in active_optionals)]
            while inp_dims and sig_dims:
                inp_dim = inp_dims.pop()
                sig_dim = sig_dims.pop()
                if sig_dim.name in dim_map:
                    if inp_dim != dim_map[sig_dim.name]:
                        break_flag = True
                        break
                else:
                    dim_map[sig_dim.name] = inp_dim
            else:
                if not inp_dims: # not inp_dims
                    # The rest of the signature dims must be an inactive optional and not already in the dim_map
                    for sig_dim in sig_dims:
                        if sig_dim.name in optional_dim_names or sig_dim.name in dim_map:
                            # Try another set of optionals
                            break_flag = True
                            break
            if break_flag:
                break
        if break_flag:
            continue
        return active_optionals
    raise ValueError(f"The input dimensions {input_dims} are not compatible with the signature {signature}.")


def concretize_signature(signature: UfuncSignature, *input_dims: List[Dimension]) -> UfuncSignature:
    """
    Returns the signature with its inactive optionals removed and its active optionals replaced with non-optionals.
    """
    optional_set = compute_active_signature_optionals(signature, *input_dims)
    return signature.concretise_optionals(optional_set)


def get_dim_map(signature: UfuncSignature, *input_dims: List[Dimension]) -> Dict[str, Dimension]:
    """
    Returns a dictionary mapping the names of the active dimensions of the given signature to the names of the corresponding input dimensions.
    """
    concrete_signature = concretize_signature(signature, *input_dims)
    return {sig_dim.name: inp_dim for sig_dims, inp_dims in zip(concrete_signature.input_dims, input_dims) for sig_dim, inp_dim in zip(reversed(sig_dims.dims), reversed(inp_dims))}


def compute_noncore_dims(signature: UfuncSignature, *input_dims: List[Dimension]) -> List[Dimension]:
    """
    Returns the noncore input and output dimensions of the given signature with the given input dimensions.
    """
    concrete_signature = concretize_signature(signature, *input_dims)
    dim_map = get_dim_map(concrete_signature, *input_dims)
    input_noncore_dims = [[dim_map[sig_dim.name] for sig_dim in sig_dims.dims] for sig_dims in concrete_signature.input_dims]
    output_noncore_dims = [tuple(dim_map[sig_dim_part.name] for sig_dim_part in sig_dim) if isinstance(sig_dim, tuple) else dim_map[sig_dim.name]  for sig_dim in concrete_signature.output_dims.dims]
    return input_noncore_dims, output_noncore_dims


def compute_core_dims(signature: UfuncSignature, *input_dims: List[Dimension]) -> List[Dimension]:
    """
    Returns the core input dimensions of the given signature with the given input dimensions.
    """
    input_noncore_dims, _ = compute_noncore_dims(signature, *input_dims)
    input_core_dims = [dims[:-len(noncore_dims)] if noncore_dims else dims for dims, noncore_dims in zip(input_dims, input_noncore_dims)]
    return input_core_dims


def calculate_output_dims_from_signature(signature: UfuncSignature, *input_dims: List[Dimension]) -> List[Dimension]:
    """
    Returns the dimensions of the output of the given signature with the given input dimensions.
    """
    input_noncore_dims, output_noncore_dims = compute_noncore_dims(signature, *input_dims)
    input_core_dims = [dims[:-len(noncore_dims)] if noncore_dims else dims for dims, noncore_dims in zip(input_dims, input_noncore_dims)]
    aligned_core_dims = get_final_aligned_dims(*input_core_dims)
    return [*aligned_core_dims, *output_noncore_dims]


@v_args(inline=True)
class TreeToDimsDeclaration(Transformer):
    def dims(self, *dims) -> Tuple[Dimension]:
        return tuple(dim for dim in dims if dim is not None)
    
    def dim(self, value, *x) -> Union[Dimension, Tuple]:
        print(x)
        return value
    
    def tuple(self, *dims) -> Tuple[Dimension]:
        return tuple(dim for dim in dims if dim is not None)
    
    def sep(self) -> None:
        return None

    def NAME(self, tree) -> Dimension:
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
    def dims(self, *dims) -> Tuple[Dimension]:
        return tuple(dim for dim in dims if dim is not None)
    
    def dim(self, value, *x) -> Union[Dimension, Tuple]:
        print(x)
        return value
    
    def tuple(self, *dims) -> Tuple[Dimension]:
        return tuple(dim for dim in dims if dim is not None)
    
    def sep(self) -> None:
        return None

    def NAME(self, tree) -> Dimension:
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
    if isinstance(dims, Dimension):
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
    if isinstance(dims, Dimension):
        return (dims,)
    else:
        return dims
