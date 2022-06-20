import typing
from typing import *
import string
import re
import math
from itertools import zip_longest
from itertools import chain, combinations

from .parse_numpy_ufunc_signature import parse_ufunc_signature
from .dim_calcs import broadcast_dims, compute_noncore_dims, dims_are_aligned, parse_dims, calculate_output_dims_from_signature
from .raw_ops import align_to_dims, align_arrays
from .types import ArrayLike, ConcreteArrayLike, Dimension, LazyArrayLike, RawArrayLike

import numpy as np
import numpy.typing as npt


class Lazy:
    pass


def reduce_sum(a: ConcreteArrayLike, dims: Union[Container[Dimension], Iterator[Dimension]]) -> ConcreteArrayLike:
    """
    Collapse the given array along the given dimensions.
    """
    if not set(a.dims) & set(dims):
        return a
    if set(dims) - set(a.dims):
        raise ValueError(f"The dimensions {dims} must be a subset of the dimensions {a.dims} of the input array.")
    return einarray(np.sum(a.__array__(), axis=tuple(i for i, dim in enumerate(a.dims) if dim in dims)), [dim for dim in a.dims if dim not in dims])


class lazy_ufunc(LazyArrayLike, np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, ufunc: Callable, method: str, *inputs, **kwargs):
        self.ufunc = ufunc
        self.method = method
        self.inputs = inputs
        self.kwargs = kwargs
        
    def get_dims_unordered(self) -> Set[Dimension]:
        return {dim for inp in self.inputs if hasattr(inp, "get_dims_unordered") for dim in inp.get_dims_unordered()}

    def coerce(self, dims: List[Dimension], do_not_collapse: Set[Dimension], force_align: bool = True) -> ConcreteArrayLike:
        """
        Coerce the lazy array to a concrete array of the given dimensions, ignoring dimensions that do not appear in self.inputs and collapsing those that don't appear in either dims or do_not_collapse.
        """
        if self.ufunc in [np.add, np.subtract]:
            # Coerce the inputs into the same dimensions.
            inputs = [inp.coerce(dims, do_not_collapse, force_align=False) for inp in self.inputs]
            # Collapse all dimensions except those in contained in dims or do_not_collapse.
            dims_to_collapse = {dim for inp in inputs for dim in inp.dims} - set(dims) - do_not_collapse
            if dims_to_collapse:
                inputs = [reduce_sum(inp, set(inp.dims) & dims_to_collapse) for inp in inputs]
            # Return the result.
            signature = parse_ufunc_signature(self.ufunc.signature or ','.join(['()']*len(inputs)) + '->()')
            raw_input_arrays, output_dims = align_arrays(*inputs, signature=signature, return_output_dims=True)
            out = einarray(getattr(self.ufunc, self.method)(*raw_input_arrays, **self.kwargs), output_dims)
            if force_align and not dims_are_aligned(dims, output_dims):
                # Align the output to the given dimensions.
                return out[[dim for dim in dims if dim in out.dims]]
            else:
                return out
        elif self.ufunc in [np.multiply, np.divide]:
            # Prevent collapsing along dimensions shared by two or more inputs.
            dims_used_at_least_once = set()
            dims_used_at_least_twice = set()
            for inp in self.inputs:
                dims_used_at_least_twice |= inp.get_dims_unordered() & dims_used_at_least_once
                dims_used_at_least_once |= inp.get_dims_unordered()
            # Coerce each input
            inputs = [inp.coerce(dims, do_not_collapse | dims_used_at_least_twice, force_align=False) for inp in self.inputs]
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
            # TODO: the second part of this may result in needless transpositions. To fix, try to make output_dims more like the shape of the inputs.
            out_dims = [dim for dim in dims if dim in all_input_dims] + [dim for dim in all_input_dims & do_not_collapse - set(dims)]
            signature = ",".join("".join(dim_map[dim] for dim in inp.dims) for inp in inputs) + "->" + "".join(dim_map[dim] for dim in out_dims)
            # Get the raw arrays and, if self.ufunc is a divison operation, take the reciprocal of the second input
            if self.ufunc in [np.divide]:
                assert len(inputs) == 2
                inputs[1] = np.reciprocal(inputs[1])
            # Return the result
            return einarray(np.einsum(signature, *inputs, **self.kwargs), [dim for dim in out_dims])
        else:
            raise NotImplementedError(f"The lazy ufunc {self.ufunc} is not implemented.")
        
    def __getitem__(self, dims) -> ConcreteArrayLike:
        return self.coerce(dims, set())

    def __array_ufunc__(self, ufunc: Callable, method: str, *inputs, **kwargs) -> LazyArrayLike:
        if ufunc in [np.add, np.subtract, np.multiply, np.divide]:
            return lazy_ufunc(ufunc, method, *inputs, **kwargs)
        else:
            # Use einarray.__array_ufunc__
            return einarray.__array_ufunc__(self, ufunc, method, *inputs, **kwargs)
    
    def __array__(self, dtype: Optional[npt.DTypeLike] = None) -> ConcreteArrayLike:
        return NotImplemented

    def __repr__(self) -> str:
        return f"lazy_ufunc({self.ufunc}, {self.method}, {self.inputs}, {self.kwargs})"


class einarray(ConcreteArrayLike, np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, a: ConcreteArrayLike, dims: List[Dimension]) -> None:
        if isinstance(a, (int, float, complex, np.number)):
            a = np.array(a)
        # if not isinstance(a, ConcreteArrayLike):
        if not isinstance(a, (RawArrayLike, ConcreteArrayLike)):
            raise ValueError(f"The array must be a concrete array. Got {a} of type {type(a)}.")
        if a.ndim != len(dims):
            raise ValueError(f"The number {a.ndim} of dimensions in the array does not match the number {len(dims)} of dimensions passed to the constructor.")
        self.a = a
        self.dims = tuple(dims)
        
    def get_dims_unordered(self) -> Set[Dimension]:
        return set(self.dims)
    
    def coerce(self, dims: List[Dimension], do_not_collapse: Set[Dimension], force_align: bool = True) -> ConcreteArrayLike:
        # Collapse all dimensions except those in contained in dims or do_not_collapse.
        dims_to_collapse = set(self.dims) - set(dims) - do_not_collapse
        out = reduce_sum(self, dims_to_collapse)
        if not force_align:
            return out
        else:
            out_dims = [dim for dim in dims if dim in out.dims] + [dim for dim in out.dims if dim not in dims and dim in do_not_collapse]
            return einarray(align_to_dims(out, out_dims), out_dims)
        
    def __getitem__(self, dims: List[Dimension]) -> ConcreteArrayLike:
        return self.coerce(dims, set())
    
    def __array_ufunc__(self, ufunc: Callable, method: str, *inputs, **kwargs) -> ConcreteArrayLike:
        if ufunc in [np.add, np.subtract, np.multiply, np.divide]:
            # Execute lazily
            # There may be oppertunities to reduce terms before addition or subtraction.
            # Multiplication and multiplication can be done efficiently using an einsum, but we need to know for sure what the output dims are before we can do that.
            return lazy_ufunc(ufunc, method, *inputs, **kwargs)
        else:
            # Force lazy_ufunc inputs to be concrete arrays without collapsing any dimensions.
            all_dims = {dim for inp in inputs for dim in inp.get_dims_unordered()}
            inputs = [inp.coerce([], all_dims, force_align=False) for inp in inputs]
            # Align input arrays
            signature = parse_ufunc_signature(ufunc.signature or ','.join(['()']*len(inputs)) + '->()')
            raw_arrays, output_dims = align_arrays(*inputs, signature=signature, return_output_dims=True)
            # Apply the ufunc to the raw arrays.
            return einarray(getattr(ufunc, method)(*raw_arrays, **kwargs), dims=output_dims)
    def __array__(self, dtype: Optional[npt.DTypeLike] = None) -> ConcreteArrayLike:
        return self.a

    def __repr__(self) -> str:
        return f"einarray({self.a}, {self.dims})"