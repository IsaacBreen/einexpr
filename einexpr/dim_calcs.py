from itertools import zip_longest
from typing import List, Tuple, Sequence, Any

from einexpr.types import Dimension
from .utils import powerset
from .parse_numpy_ufunc_signature import UfuncSignature, UfuncSignatureDimensions, parse_ufunc_signature
from .types import ConcreteArrayLike, RawArray


def broadcast_dims(*ein_dims: Sequence[Dimension]) -> List[Dimension]:
    """
    Returns the broadcast of the given dimensions.
    """
    broad = max(ein_dims, key=len)
    for dims in ein_dims:
        if tuple(broad[-len(dims):]) != tuple(dims):
            raise ValueError(f"The dimensions {dims} are not broadcastable to {broad}.")
    return broad


def dims_after_alignment(*ein_dims: Sequence[Dimension]) -> List[Dimension]:
    """
    Returns a single list of dimensions that contains all the dimensions of the given inputs.
    """
    aligned = max(ein_dims, key=len)
    for dims in ein_dims:
        for dim in dims:
            if dim not in aligned:
                aligned.append(dim)
    return aligned


def calculate_transpose(dims_from: Sequence[Dimension], dims_to: Sequence[Dimension]) -> List[Dimension]:
    """
    Returns lists of the transpositions required to align to given dimensions. Broadcast dimensions - those
    that will need to be expanded - are represented by None.
    """
    return [dims_from.index(dim) if dim in dims_from else None for dim in dims_to]


def calculate_output_dims_from_signature_and_einarrays(signature: UfuncSignature, *inputs: ConcreteArrayLike) -> List[str]:
    """
    The noncore dimensions of the input arrays must be broadcastable.
    """
    dim_map = {}
    noncore_dims_rev = []
    optional_dim_names = {dim.name for dims in (signature.output_dims, *signature.input_dims) for dim in dims.dims if dim.optional}
    for optional_set in powerset(optional_dim_names):
        break_flag = False
        for inp, ufunc_inp_sigs in zip_longest(inputs, signature.input_dims):
            inp_dims = list(inp.dims)
            sig_dims = [dim for dim in ufunc_inp_sigs.dims if dim.name not in optional_set]
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
                if not sig_dims:
                    # The rest of the input dims are non-core
                    for inp_dim, noncore_dim in zip(inp_dims, noncore_dims_rev):
                        if inp_dim != noncore_dim:
                            break_flag = True
                            break
                    if len(inp_dims) > len(noncore_dims_rev):
                        noncore_dims_rev = inp_dims
                else: # not inp_dims
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
        core_dims = [dim_map[dim.name] for dim in signature.output_dims.dims if dim.name not in optional_dim_names]
        output_dims = noncore_dims_rev + core_dims
        return output_dims
    raise ValueError(f"The input {inputs} is not compatible with the signature {signature}.")


def get_core_dims(signature_part: UfuncSignatureDimensions, *dims: List[Dimension]) -> List[Dimension]:
    """
    Returns the core dimensions of the given signature.
    """
    return dims[:-len(signature_part.dims)]


def get_noncore_dims(signature_part: UfuncSignatureDimensions, *dims: List[Dimension]) -> List[Dimension]:
    """
    Returns the noncore dimensions of the given signature.
    """
    return dims[-len(signature_part.dims):]


def split_core_dims(signature_part: UfuncSignatureDimensions, *dims: List[Dimension]) -> Tuple[List[Dimension], List[Dimension]]:
    """
    Returns the core and noncore dimensions respectively of the given signature.
    """
    return dims[:-len(signature_part.dims)], dims[-len(signature_part.dims):]
