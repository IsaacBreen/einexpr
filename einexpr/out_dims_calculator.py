from regex import R
import einexpr

from .base_typing import EinarrayLike
from .einarray_utils import *

Dimension = str

backend_array_types = dict()
backend_dim_kwargs_to_resolve = ['dim', 'axis']

function_registry = []


class DimensionCalculator:
    pass


class SingleArgumentElementwise:
    @staticmethod
    def validate_args(args, kwargs):
        assert isinstance(args, (list, tuple))
        assert isinstance(kwargs, dict)
        assert isinstance(args[0], EinarrayLike)
        assert all(not isinstance(arg, EinarrayLike) for arg in args[1:])
        assert all(not isinstance(arg, EinarrayLike) for arg in kwargs.values())
    
    @staticmethod
    def process_args(args, kwargs):
        return (args[0].a, *args[1:]), kwargs
    
    @staticmethod
    def calculate_output_dims(args, kwargs):
        return args[0].dims
    
    @staticmethod
    def calculate_output_ambiguous_dims(args, kwargs):
        return {dim for ambiguous_dims in args[0].ambiguous_dims for dim in ambiguous_dims}


class MultiArgumentElementwise:
    @staticmethod
    def validate_args(args, kwargs):
        assert isinstance(args, (list, tuple))
        assert isinstance(kwargs, dict)
        assert all(isinstance(arg, EinarrayLike) for arg in args)
        assert all(not isinstance(arg, EinarrayLike) for arg in kwargs.values())
    
    @staticmethod
    def process_args(args, kwargs):
        raw_aligned_arrays = einexpr.align_arrays(*args)
        return raw_aligned_arrays, kwargs

    @staticmethod
    def calculate_output_dims(args, kwargs):
        return einexpr.get_final_aligned_dims(*(arg.dims for arg in args))
    
    @staticmethod
    def calculate_output_ambiguous_dims(args, kwargs):
        ambiguous_dims = einexpr.calculate_ambiguous_final_aligned_dims(*(arg.dims for arg in args))
        ambiguous_dims |= {dim for arg in args for dim in arg.ambiguous_dims}
        return ambiguous_dims


class MultiDimensionReduction:
    @staticmethod
    def validate_args(args, kwargs):
        # TODO:
        # - Check that axes are in the array
        # - raise error when there are duplicate axes, esp of different types (Dimension, int, and negative int)
        assert isinstance(args, (list, tuple))
        assert isinstance(kwargs, dict)
        assert isinstance(args[0], EinarrayLike)
        assert all(not isinstance(arg, EinarrayLike) for arg in args[1:]) # TODO: mapreduce over pytree
        assert all(not isinstance(arg, EinarrayLike) for arg in kwargs.values())
        assert 'axis' in args        
    
    @staticmethod
    def _calculate_axis(args, kwargs):
        axis = kwargs.get('axis')
        if axis is None:
            axis = list(range(len(args[0].dims)))
        elif isinstance(axis, (Dimension, int)):
            axis = [axis]
        axis = [args[0].dims.index(dim) if isinstance(dim, Dimension) else dim for dim in axis]
        assert all(isinstance(dim, int) for dim in axis)
        axis = [i if i >= 0 else i + len(args[0].dims) for i in axis]
        return tuple(axis)
    
    @staticmethod
    def process_args(args, kwargs):
        axis = MultiDimensionReduction._calculate_axis(args, kwargs)
        return (args[0].a, *args[1:]), {**kwargs, 'axis': axis}

    @staticmethod
    def calculate_output_dims(args, kwargs):
        axis = MultiDimensionReduction._calculate_axis(args, kwargs)
        return [dim for i, dim in enumerate(args[0].dims) if i not in axis]
    
    @staticmethod
    def calculate_output_ambiguous_dims(args, kwargs):
        axis = MultiDimensionReduction._calculate_axis(args, kwargs)
        ambiguous_out_dims = args[0].ambiguous_dims - {dim for i, dim in enumerate(args[0].dims) if i in axis}
        return ambiguous_out_dims


class SingleDimensionReduction(MultiDimensionReduction):
    @staticmethod
    def validate_args(args, kwargs):
        if 'axis' in kwargs and isinstance(kwargs['axis'], (tuple, list)):
            raise ValueError("Multiple axes not supported")
        elif 'dims' in kwargs and isinstance(kwargs['dims'], (tuple, list)):
            raise ValueError("Multiple axes not supported")
        MultiDimensionReduction.validate_args(args, kwargs)


class Concatenation:
    @staticmethod
    def validate_args(args, kwargs):
        assert isinstance(args, (list, tuple))
        assert isinstance(kwargs, dict)
        assert isinstance(args[0], (list, tuple)) and all(isinstance(arg, EinarrayLike) for arg in args[0])
        assert all(not isinstance(arg, EinarrayLike) for arg in args[1:])
        if kwargs['axis'] is None:
            raise ValueError("Using ``axis=None`` (flattening concatenation) is not yet supported")
    
    @staticmethod
    def _calculate_axes(args, kwargs):
        axes = kwargs['axis']
        if isinstance(axes, (Dimension, int)):
            axes = [axes] * len(args[0])
        if not isinstance(axes, (list, tuple)):
            raise ValueError("Axes must be a list or tuple")
        axes = list(axes)
        for i, (axis, arg) in enumerate(zip(axes, args[0])):
            # If the axis is an integer, convert it to a Dimension
            if isinstance(axis, int):
                axis = arg.dims[axis]
                axes[i] = axis
            elif not isinstance(axis, Dimension):
                raise ValueError(f"Invalid axis {axis}")
            assert axis not in arg.ambiguous_dims
        return axes

    @staticmethod
    def process_args(args, kwargs):
        axes = Concatenation._calculate_axes(args, kwargs)
        axis_num = args[0][0].dims.index(axes[0])
        # Align the arrays. Use the shape of the first array as the template.
        aligned_args = [args[0][0]]
        for axis, arg in zip(axes[1:], args[0][1:]):
            aligned_args.append(arg[tuple(dim if dim != axes[0] else axis for dim in args[0][0].dims)])
        raw_aligned_args = [arg.a for arg in aligned_args]
        return (raw_aligned_args, *args[1:]), {**kwargs, 'axis': axis_num}

    @staticmethod
    def calculate_output_dims(args, kwargs):
        axes = Concatenation._calculate_axes(args, kwargs)
        # Check that arrays share all dimensions except the concatenated ones
        out_dims_set = set(args[0][0].dims) - {axes[0]}
        for axis, arg in zip(axes[1:], args[0][1:]):
            arg_dims_set = set(arg.dims) - {axis}
            if arg_dims_set != out_dims_set:
                raise ValueError(f"Arrays must have all the same dimensions except those concatentated over. The first input array {args[0][0]} has non-concatenated dimensions {out_dims_set}, while the input array {arg} has non-concatenated dimensions {arg_dims_set}.")
        out_dims = [dim if dim != axes[0] else tuple(axes) for dim in args[0][0].dims]
        return out_dims
        
    @staticmethod
    def calculate_output_ambiguous_dims(args, kwargs):
        axes = Concatenation._calculate_axes(args, kwargs)
        ambiguous_dims = {dim for arg in args[0] for dim in arg.ambiguous_dims if dim not in axes}
        for axis, arg in zip(axes[1:], args[0][1:]):
            for dim0, dim in zip(args[0][0].dims, arg.dims):
                if axes[0] != dim0 and dim != axis and dim0 != dim:
                    ambiguous_dims.add(dim)
        return ambiguous_dims
