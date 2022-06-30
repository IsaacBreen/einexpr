"""
Definitions:
- Ambiguous dimension: A dimensions that may occur one of many positions in the array.
- Core dimension: (numpy) A dimension that is not in the 
"""

import math
import re
import string
import typing
from itertools import chain, combinations, zip_longest
from typing import *

import numpy as np
import numpy.typing as npt
import torch as pt

from .backends import *
from .dim_calcs import *
from .exceptions import *
from .parse_numpy_ufunc_signature import make_empty_signature_str, parse_ufunc_signature
from .raw_ops import align_arrays, align_to_dims
from .typing import *


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


def process_inp(inp):
    if isinstance(inp, DimensionlessLike):
        return einarray(inp)
    else:
        return inp

class lazy_func(LazyArrayLike, np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, func: Callable, signature: UfuncSignature, *inputs, **kwargs):
        self.func = func
        self.signature = signature
        self.inputs = [process_inp(inp) for inp in inputs]
        self.kwargs = kwargs
    
    @property
    def dims(self) -> List[Dimension]:
        """
        Return the dimensions of the lazy_func.
        """
        return calculate_output_dims_from_signature(self.signature, *(inp.dims for inp in self.inputs))

    @property
    def ambiguous_dims(self) -> Set[Dimension]:
        # Align input arrays
        # Collect ambiguous dimensions. Assume non-core dimensions are unambiguous.
        inp_core_dims = compute_core_dims(self.signature, *(inp.dims for inp in self.inputs))
        ambiguous_dims = calculate_ambiguous_get_final_aligned_dims(*inp_core_dims)
        ambiguous_dims |= {dim for inp in self.inputs if isinstance(inp, ArrayLike) for dim in inp.ambiguous_dims}
        return ambiguous_dims

    def get_dims_unordered(self) -> Set[Dimension]:
        return {dim for inp in self.inputs if hasattr(inp, "get_dims_unordered") for dim in inp.get_dims_unordered()}

    def coerce(self, dims: List[Dimension], do_not_collapse: Set[Dimension] = None, force_align: bool = True, ambiguous_dims: Set[Dimension] = None) -> ConcreteArrayLike:
        """
        Coerce the lazy array to a concrete array of the given dimensions, ignoring dimensions that do not appear in self.inputs and collapsing those that don't appear in either dims or do_not_collapse.
        """
        do_not_collapse = do_not_collapse or set()
        ambiguous_dims = ambiguous_dims or set()
        if self.func in [np.add.__call__, np.subtract.__call__]:
            # Coerce the inputs into the same dimensions.
            inputs = [inp.coerce(dims, do_not_collapse, force_align=False) for inp in self.inputs]
            # Collapse all dimensions except those in contained in dims or do_not_collapse.
            dims_to_collapse = {dim for inp in inputs for dim in inp.dims} - set(dims) - do_not_collapse
            if dims_to_collapse:
                inputs = [reduce_sum(inp, set(inp.dims) & dims_to_collapse) for inp in inputs]
            # Return the result.
            raw_input_arrays, output_dims = align_arrays(*inputs, signature=self.signature, return_output_dims=True)
            out = einarray(self.func(*raw_input_arrays, **self.kwargs), output_dims, ambiguous_dims=ambiguous_dims)
            if force_align and not dims_are_aligned(dims, output_dims):
                # Align the output to the given dimensions.
                return out[[dim for dim in dims if dim in out.dims]]
            else:
                return out
        elif self.func in [np.multiply.__call__, np.divide.__call__, np.true_divide.__call__, ]:
            # Prevent collapsing along dimensions shared by two or more inputs.
            all_input_dims = {dim for inp in self.inputs for dim in inp.dims}
            # Coerce each input
            inputs = [inp.coerce(dims, do_not_collapse | all_input_dims, force_align=False) for inp in self.inputs]
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
            # Get the raw arrays and, if self.func is a divison operation, take the reciprocal of the second input
            if self.func in [np.divide.__call__, np.true_divide.__call__]:
                assert len(inputs) == 2
                inputs[1] = np.reciprocal(inputs[1])
            # Return the result
            return einarray(np.einsum(signature, *(inp.a for inp in inputs), **self.kwargs), [dim for dim in out_dims], ambiguous_dims=ambiguous_dims)
        else:
            raise NotImplementedError(f"Lazy evaluation for {self.func} is not yet implemented.")
        
    @property
    def a(self) -> RawArrayLike:
        return self.coerce(self.dims).a
        
    def __getitem__(self, dims) -> ConcreteArrayLike:
        return self.coerce(parse_dims(dims), set())

    def __einarray_function__(self, func: Callable, signature: UfuncSignature, *inputs, **kwargs) -> ArrayLike:
        if self.func in [np.multiply.__call__, np.divide.__call__, np.true_divide.__call__, ] and func in [np.add.__call__, np.subtract.__call__, np.multiply.__call__, np.divide.__call__, np.true_divide.__call__, ]:
            return lazy_func(func, signature, *inputs, **kwargs)
        elif self.func in [np.add.__call__, np.subtract.__call__] and func in [np.add.__call__, np.subtract.__call__]:
            return lazy_func(func, signature, *inputs, **kwargs)
        else:
            return einarray.__einarray_function__(self, func, signature, *[inp.coerce(inp.dims, ambiguous_dims=inp.ambiguous_dims) if isinstance(inp, lazy_func) else inp for inp in inputs], **kwargs)
    
    def __array_ufunc__(self, ufunc: Callable, method: str, *inputs, **kwargs) -> LazyArrayLike:
        signature_str = ufunc.signature or ','.join(['()']*len(inputs)) + '->()'
        signature = parse_ufunc_signature(signature_str)
        return self.__einarray_function__(getattr(ufunc, method), signature, *inputs, **kwargs)

    def __array__(self, dtype: Optional[npt.DTypeLike] = None) -> ConcreteArrayLike:
        return self.a
    
    def __bool__(self) -> bool:
        return bool(self.a)
    
    # JAX support
    def _tree_flatten(self):
        children = (self.inputs,)
        aux_data = {'func': self.func, 'signature': self.signature, 'kwargs': self.kwargs}
        return (children, aux_data)
    
    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    def __repr__(self) -> str:
        return f"lazy_func({self.func}, {self.signature}, {self.inputs}, {self.kwargs})"


class einarray(ConcreteArrayLike, np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, a: Union[RawArrayLike, ConcreteArrayLike], dims: List[Dimension] = None, ambiguous_dims: Set[Dimension] = None, copy: bool=True, backend: Literal["numpy", "torch"] = None) -> None:
        self.a = a
        self.dims = parse_dims(dims)
        self.ambiguous_dims = ambiguous_dims or set()
        if isinstance(self.a, LazyArrayLike):
            self.a = self.a.coerce(self.dims, self.a.ambiguous_dims)
        if isinstance(self.a, ArrayLike):
            if self.dims != self.a.dims:
                raise ValueError(f"The dimensions passed to the constructor must match the dimensions of the concrete array. Got {self.dims} but the array has dimensions {self.a.dims}.")
            self.a = self.a.a
        # Convert self.a to the specified backend
        if backend is None:
            if isinstance(self.a, RawArrayLike):
                # Just use raw array as-is
                self.a = self.a
            else:
                # The user may have passed a list array (e.g. a=[[1,2],[3,4]]). In this case, we hope that self.a can be converted into np.array.
                self.a = np.array(self.a, copy=copy)
                # If self.a is dimensionless (a value with a shape of () - generally an int or a float), we require the dimensions to be specified explicitly.
                assert self.dims is not None, f"The dimensions must be specified explicitly when converting a {type(self.a).__name__} to a numpy array."
        elif backend == "numpy":
            self.a = np.array(self.a, copy=copy)
        elif backend == "torch":
            self.a = pt.asarray(self.a, copy=copy)
        else:
            raise ValueError(f"The backend must be either 'numpy' or 'torch'. Got {backend}.")
        if isinstance(self.a, DimensionlessLike) or isinstance(self.a, RawArrayLike) and not self.a.shape:
            assert not self.dims, f"The dimensions of a dimensionless value must be empty or None. Got {self.dims}."
            self.dims = ()
        elif not isinstance(self.dims, (tuple, list)):
            if dims == self.dims:
                raise ValueError(f"The dimensions passed to the constructor must be a list or tuple. Got value {dims!r} of type {type(dims).__name__}.")
            else:
                raise ValueError(f"Possible internal error. Dimensions must be a list or tuple. The value of the dims argument is {dims!r} of type {type(dims).__name__}, and {self.dims!r} of type {type(self.dims).__name__} was inferred.")
        if isinstance(self.a, RawArrayLike) and self.a.ndim != len(self.dims):
            raise ValueError(f"The number {self.a.ndim} of dimensions in the array does not match the number {len(self.dims)} of dimensions passed to the constructor.")
        if not self.ambiguous_dims.issubset(self.dims):
            raise ValueError(f"The ambiguous dimensions {self.ambiguous_dims} are must be a subset of the dimensions {self.dims}.")

    @property
    def backend(self) -> str:
        detect_backend(self.a)
        
    def get_dims_unordered(self) -> Set[Dimension]:
        return set(self.dims)
    
    def coerce(self, dims: List[Dimension], do_not_collapse: Set[Dimension], force_align: bool = True, ambiguous_dims: Set[Dimension] = None) -> ConcreteArrayLike:
        ambiguous_dims = ambiguous_dims or set()
        # Collapse all dimensions except those in contained in dims or do_not_collapse.
        dims_to_collapse = set(self.dims) - set(dims) - do_not_collapse
        out = reduce_sum(self, dims_to_collapse)
        if not force_align:
            return out
        else:
            out_dims = [dim for dim in dims if dim in out.dims] + [dim for dim in out.dims if dim not in dims and dim in do_not_collapse]
            return einarray(align_to_dims(out, out_dims), out_dims, ambiguous_dims=ambiguous_dims)
        
    def __getitem__(self, dims: List[Dimension]) -> ConcreteArrayLike:
        return self.coerce(parse_dims(dims), set())
    
    def __einarray_function__(self, func: Callable, signature: UfuncSignature, *inputs, **kwargs) -> ArrayLike:
        if func in [np.add.__call__, np.subtract.__call__, np.multiply.__call__, np.divide.__call__, np.true_divide.__call__, ]:
            # Execute lazily
            # There may be oppertunities to reduce terms before addition or subtraction.
            # Multiplication and multiplication can be done efficiently using an einsum, but we need to know for sure what the output dims are before we can do that.
            return lazy_func(func, signature, *inputs, **kwargs)
        else:
            inputs = [process_inp(inp) for inp in inputs]
            inputs_dims = [inp.dims for inp in inputs]
            # Exchange dimension name for index if necessary.
            for kw in backend_dim_kwargs_to_resolve:
                if kw in kwargs:
                    if kwargs[kw] in self.ambiguous_dims:
                        raise AmbiguousDimensionException(f"The position of the dimension {kwargs[kw]} is ambiguous.")
                    if isinstance(kwargs[kw], Dimension):
                        kwargs[kw] = self.dims.index(kwargs[kw])
                    elif isinstance(kwargs[kw], (list, tuple)):
                        kwargs[kw] = [self.dims.index(dim) for dim in kwargs[kw]]
                    elif not isinstance(kwargs[kw], int):
                        raise ValueError(f"The value of the {kw} argument must be a dimension or a list of dimensions. Got {kwargs[kw]} of type {type(kwargs[kw]).__name__}.")
            # Force lazy_func inputs to be concrete arrays without collapsing any dimensions.
            all_dims = {dim for inp in inputs for dim in inp.get_dims_unordered()}
            # Collect ambiguous dimensions
            inp_core_dims = compute_core_dims(signature, *inputs_dims)
            ambiguous_dims = calculate_ambiguous_get_final_aligned_dims(*inp_core_dims)
            ambiguous_dims |= {dim for inp in inputs if isinstance(inp, ArrayLike) for dim in inp.ambiguous_dims}
            # All potential noncore dimensions must be unambiguous. (This is an assumption of calculate_output_dims_from_signature.)
            inp_potential_noncore_dims = [dims[-len(inp_sig):] if len(inp_sig) > 0 else [] for dims, inp_sig in zip(inputs_dims, signature.input_dims)]
            all_potential_noncore_dims = {dim for dims in inp_potential_noncore_dims for dim in dims}
            if ambiguous_dims & all_potential_noncore_dims:
                raise AmbiguousDimensionException(f"Potentially noncore dimensions must be unambiguous.")
            # Align input arrays
            inputs = [inp.coerce([], all_dims, force_align=False, ambiguous_dims=inp.ambiguous_dims) for inp in inputs]
            raw_arrays, output_dims = align_arrays(*inputs, signature=signature, return_output_dims=True)
            # Apply the func to the raw arrays.
            return einarray(func(*raw_arrays, **kwargs), dims=output_dims, ambiguous_dims=ambiguous_dims)

    def __array_ufunc__(self, ufunc: Callable, method: str, *inputs, **kwargs) -> ConcreteArrayLike:
        signature_str = ufunc.signature or ','.join(['()']*len(inputs)) + '->()'
        signature = parse_ufunc_signature(signature_str)
        return self.__einarray_function__(getattr(ufunc, method), signature, *inputs, **kwargs)
    
    def __array_function__(self, func: Callable, types: List[Type], args: List[Any], kwargs: Dict[str, Any]) -> ConcreteArrayLike:
        registration = next(search_function_registry(function=func), None)
        if registration is None:
            raise ValueError(f"The function {func} is not registered.")
        if registration.signature is None:
            signature = make_empty_signature_str(len(args))
        elif isinstance(registration.signature, str):
            signature = registration.signature
        else:
            signature = registration.signature(args, kwargs)
        signature = parse_ufunc_signature(signature)
        if all(isinstance(arg, ArrayLike) for arg in args):
            return self.__einarray_function__(func, signature, *args, **kwargs)
        else:
            if len(args) == 1 and isinstance(args[0][0], ArrayLike):
                return self.__einarray_function__(lambda *args, **kwargs: func(args, **kwargs), signature, *args[0], **kwargs)
            else:
                raise ValueError(f"The arguments must be either arrays or a tuple of arrays. Got {args}.")
        
    def __array__(self, dtype: Optional[npt.DTypeLike] = None) -> ConcreteArrayLike:
        return self.a
    
    def __bool__(self) -> bool:
        return bool(self.a)

    # JAX support
    def _tree_flatten(self):
        children = (self.a,)
        aux_data = {'dims': self.dims, 'ambiguous_dims': self.ambiguous_dims}
        return (children, aux_data)
    
    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    def __repr__(self) -> str:
        return f"einarray({self.a}, {self.dims})"
    
    
from jax import tree_util

tree_util.register_pytree_node(lazy_func, lazy_func._tree_flatten, lazy_func._tree_unflatten)
tree_util.register_pytree_node(einarray, einarray._tree_flatten, einarray._tree_unflatten)
