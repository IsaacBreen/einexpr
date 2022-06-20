import regex as re
from itertools import zip_longest
from typing import Dict, List, Set, Tuple, Sequence, Any


from einexpr.types import Dimension
from .utils import powerset
from .parse_numpy_ufunc_signature import UfuncSignature, UfuncSignatureDimensions, parse_ufunc_signature
from .types import ConcreteArrayLike, RawArray
from .utils import assert_outputs

def broadcast_dims(*ein_dims: List[Dimension]) -> List[Dimension]:
    """
    Returns the broadcast of the given dimensions.
    """
    broad = max(ein_dims, key=len)
    for dims in ein_dims:
        if tuple(broad[-len(dims):]) != tuple(dims):
            return
            # raise ValueError(f"The dimensions {dims} are not broadcastable to {broad}.")
    return broad


def dims_after_alignment(*ein_dims: List[Dimension]) -> List[Dimension]:
    """
    Returns a single list of dimensions that contains all the dimensions of the given inputs.
    """
    aligned = list(max(ein_dims, key=len))
    for dims in ein_dims:
        for dim in dims:
            if dim not in aligned:
                aligned.append(dim)
    return aligned


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
        break_flag = False
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
                    # The rest of the signature dims must be optional and not already in the dim_map
                    for sig_dim in sig_dims:
                        if sig_dim.name in dim_map:
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


def calculate_core_output_dims_from_signature_and_einarrays(signature: UfuncSignature, *input_dims: ConcreteArrayLike) -> List[str]:
    """
    The noncore dimensions of the input arrays must be broadcastable.
    """
    concrete_signature = concretize_signature(signature, *input_dims)
    dim_map = get_dim_map(concrete_signature, *input_dims)
    return [dim_map[sig_dim.name] for sig_dim in concrete_signature.output_dims.dims]


def compute_core_dims(signature: UfuncSignature, *input_dims: List[Dimension]) -> List[Dimension]:
    """
    Returns the core input and output dimensions of the given signature with the given input dimensions.
    """
    concrete_signature = concretize_signature(signature, *input_dims)
    dim_map = get_dim_map(concrete_signature, *input_dims)
    input_core_dims = [[dim_map[sig_dim.name] for sig_dim in sig_dims.dims] for sig_dims in concrete_signature.input_dims]
    output_core_dims = [dim_map[sig_dim.name] for sig_dim in concrete_signature.output_dims.dims]
    return input_core_dims, output_core_dims


def compute_noncore_dims(signature: UfuncSignature, *input_dims: List[Dimension]) -> List[Dimension]:
    """
    Returns the noncore input and output dimensions of the given signature with the given input dimensions.
    """
    input_core_dims, _ = compute_core_dims(signature, *input_dims)
    input_noncore_dims = [dims[:-len(core_dims)] if core_dims else dims for dims, core_dims in zip(input_dims, input_core_dims)]
    return input_noncore_dims


def calculate_signature_transexpands(signature: UfuncSignature, *input_dims: List[Dimension]) -> List[Dimension]:
    """
    Returns the transpositions and expansions required to align to the core dimensions of the given signature with the given input dimensions.
    """
    input_core_dims, output_core_dims = compute_core_dims(signature, *input_dims)
    input_noncore_dims = [dims[:-len(core_dims)] if core_dims else dims for dims, core_dims in zip(input_dims, input_core_dims)]
    aligned_noncore_dims = dims_after_alignment(*input_noncore_dims)
    input_noncore_dims_transexpand = [calculate_transexpand(noncore_dims, aligned_noncore_dims) for noncore_dims in input_noncore_dims]
    assert all(i==j for transexpand in input_noncore_dims_transexpand for j,i in enumerate(sorted(i for i in transexpand if i is not None)))
    return input_noncore_dims_transexpand


def calculate_output_dims_from_signature(signature: UfuncSignature, *input_dims: List[Dimension]) -> List[Dimension]:
    """
    Returns the dimensions of the output of the given signature with the given input dimensions.
    """
    input_core_dims, output_core_dims = compute_core_dims(signature, *input_dims)
    input_noncore_dims = [dims[:-len(core_dims)] if core_dims else dims for dims, core_dims in zip(input_dims, input_core_dims)]
    aligned_noncore_dims = dims_after_alignment(*input_noncore_dims)
    return [*aligned_noncore_dims, *output_core_dims]


def parse_dims(dims_raw: str) -> List[Dimension]:
    """
    Parses an dimensions string such as into a list of Dimension types.
    """
    if isinstance(dims_raw, str):
        dims_raw = re.split(",| ", dims_raw)
    # Parse elements of dims_raw if they are not already split into individual dimensions
    dims_raw = [dim for dims_raw_element in dims_raw for dim in re.split(",| ", dims_raw_element)]
    dims = []
    for dim in dims_raw:
        if not dim:
            continue
        if '[' in dim:
            dim = dim.split('[')
            # dims.append(NamedIndex(dim[0], dim[1][:-1]))
            raise NotImplementedError("Multiple dimension instances are not yet supported.")
        else:
            dims.append(dim)
    for dim in dims:
        if not str.isidentifier(dim):
            raise ValueError(f"The dimension {dim} is not a valid identifier.")
    return dims