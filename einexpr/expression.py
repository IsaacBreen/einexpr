import typing
from typing import *
import string
import re
import math
from itertools import zip_longest
from itertools import chain, combinations

from .parse_numpy_ufunc_signature import parse_ufunc_signature
from .dim_calcs import broadcast_dims, calculate_output_dims_from_signature_and_einarrays, get_core_dims, get_noncore_dims, split_core_dims
from .raw_ops import apply_transpose, align_to_dims, align_arrays
from .types import ConcreteArrayLike, Dimension, LazyArrayLike

import numpy as np
import numpy.typing as npt


class Lazy:
    pass


def reduce_sum(a: ConcreteArrayLike, dims: Union[Container[Dimension], Iterator[Dimension]]) -> ConcreteArrayLike:
    """
    Collapse the given array along the given dimensions.
    """
    if tuple(a.dims) == tuple(dims):
        return a
    else:
        return einarray(np.sum(a.__array__(), axis=[dim for dim in a.dims if dim in dims]), [dim for dim in a.dims if dim not in dims])


class lazy_ufunc(Lazy):
    def __init__(self, ufunc: Callable, method: str, *inputs, **kwargs):
        self.ufunc = ufunc
        self.method = method
        self.inputs = inputs
        self.kwargs = kwargs
        
    def get_dims_unordered(self) -> Set[Dimension]:
        return {dim for inp in self.inputs if hasattr(inp, "get_dims_unordered") for dim in inp.get_dims_unordered()}

    def coerce(self, dims: Sequence[Dimension], do_not_collapse: Set[Dimension]) -> ConcreteArrayLike:
        # Coerce the inputs into the given dimensions.
        if self.ufunc in [np.add, np.subtract]:
            # Coerce the inputs into the same dimensions.
            inputs = [inp.coerce(dims, do_not_collapse) for inp in self.inputs]
            # Collapse all available dimensions except the ones in do_not_collapse
            dims_to_collapse = self.get_dims_unordered() - do_not_collapse
            if dims_to_collapse:
                inputs = [reduce_sum(inp, dims_to_collapse) for inp in inputs]
            # Return the result.
            raw_arrays, output_dims = align_arrays(*inputs, return_dims=True)
            return einarray(getattr(self.ufunc, self.method)(*raw_arrays, **self.kwargs), output_dims)
        elif self.ufunc in [np.multiply, np.divide]:
            # Prevent collapsing along dimensions shared by two or more inputs.
            dims_used_at_least_once = set()
            dims_used_at_least_twice = set()
            for inp in inputs:
                dims_used_at_least_twice |= inp.get_dims_unordered() & dims_used_at_least_once
                dims_used_at_least_once |= inp.get_dims_unordered()
            # Coerce each input
            inputs = [inp.coerce(dims, do_not_collapse | dims_used_at_least_twice) for inp in inputs]
            # Create the einsum signature
            # einsum requires each dim to be represented by a single letter, so remap longer dims
            all_input_dims = {dim for inp in inputs for dim in inp.dims}
            available_letters = set(string.ascii_lowercase)
            dim_map = dict()
            for dim in all_input_dims:
                if len(dim) == 1:
                    dim_map[dim] = dim
                elif dim[0] in available_letters:
                    dim_map[dim] = dim[0]
                    available_letters.remove(dim[0])
                else:
                    dim_map[dim] = available_letters.pop()
            # Create the einsum signature
            signature = ",".join("".join(dim_map[dim] for dim in inp.dims) for inp in inputs) + "->" + "".join(dim_map[dim] for dim in dims if dim in all_input_dims)
            # Get the raw arrays and, if self.ufunc is a divison operation, take the reciprocal of the second input
            if self.ufunc in [np.divide]:
                assert len(inputs) == 2
                raw_arrays = ()
            # Return the result
            return einarray(np.einsum(signature, *inputs, **self.kwargs), [dim for dim in dims if dim in all_input_dims])

    def __array_ufunc__(self, ufunc: Callable, method: str, *inputs, **kwargs) -> LazyArrayLike:
        return lazy_ufunc(ufunc, method, *inputs, **kwargs)
        
    def __array__(self, dtype: Optional[npt.DTypeLike] = None) -> ConcreteArrayLike:
        return NotImplemented

    def __repr__(self) -> str:
        return f"lazy_ufunc({self.ufunc}, {self.method}, {self.inputs}, {self.kwargs})"


class einarray(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, a: ConcreteArrayLike, dims: Sequence[Dimension]) -> None:
        self.a = a
        self.dims = dims
        
    def get_potential(self) -> Set[Dimension]:
        return set(self.dims)
    
    def coerce(self, dims: Sequence[Dimension], do_not_collapse: Set[Dimension]) -> ConcreteArrayLike:
        return self

    def __array_ufunc__(self, ufunc: Callable, method: str, *inputs, **kwargs) -> ConcreteArrayLike:
        if ufunc in [np.add, np.subtract, np.multiply, np.divide]:
            # Execute lazily
            # There may be oppertunities to reduce terms before addition or subtraction.
            # Multiplication and multiplication can be done efficiently using an einsum, but we need to know for sure what the output dims are before we can do that.
            return lazy_ufunc(ufunc, method, *inputs, **kwargs)
        else:
            # Execute now
            signature = parse_ufunc_signature(ufunc.signature or ','.join(['()']*len(inputs)) + '->()')
            output_dims = calculate_output_dims_from_signature_and_einarrays(signature, *inputs)
            return einarray(getattr(ufunc, method)(*(inp.__array__() for inp in inputs), **kwargs), dims=output_dims)

    def __array__(self, dtype: Optional[npt.DTypeLike] = None) -> ConcreteArrayLike:
        return self.a

    def __repr__(self) -> str:
        return f"einarray({self.a}, {self.dims})"



