from click import Argument
import einexpr
from regex import R

Dimension = str

backend_array_types = dict()
backend_dim_kwargs_to_resolve = ['dim', 'axis']

function_registry = []


# TODO: can safely remove the following three methods now that proprocessing is done in the einarray magic calls
def get_dims(array):
    if isinstance(array, (int, float)):
        return ()
    elif isinstance(array, einexpr.array):
        return array.dims
    else:
        raise TypeError(f'{array} is not a recognized einexpr array')


def get_ambiguous_dims(array):
    if isinstance(array, (int, float)):
        return ()
    elif isinstance(array, einexpr.array):
        return array.ambiguous_dims
    else:
        raise TypeError(f'{array} is not a recognized einexpr array')


def get_raw(array):
    if isinstance(array, (int, float)):
        return array
    elif isinstance(array, einexpr.array):
        return array.a
    else:
        raise TypeError(f'{array} is not a recognized einexpr array')


class ArgumentHelper:    
    @staticmethod
    def preprocess_arg(arg):
            if isinstance(arg, einexpr.array):
                return arg
            elif isinstance(arg, (int, float)):
                return einexpr.array(arg, ())
            else:
                raise TypeError(f'{arg} is not a recognized einexpr array')

    @staticmethod
    def preprocess_args(args, kwargs):
        args = [ArgumentHelper.preprocess_arg(arg) for arg in args]
        kwargs = {key: ArgumentHelper.preprocess_arg(arg) for key, arg in kwargs.items()}
        return args, kwargs

    def __init__(self, args, kwargs):
        self.args, self.kwargs = ArgumentHelper.preprocess_args(args, kwargs)


class SingleArgumentElementwise:
    @staticmethod
    def validate_args(args, kwargs):
        assert isinstance(args, (list, tuple))
        assert isinstance(kwargs, dict)
        assert isinstance(args[0], einexpr.einarray)
        assert len(args) == 1
        assert all(not isinstance(arg, einexpr.einarray) for arg in kwargs.values())
    
    @staticmethod
    def process_args(args, kwargs):
        args, kwargs = ArgumentHelper.preprocess_args(args, kwargs)
        return (get_raw(args[0]), *args[1:]), kwargs
    
    @staticmethod
    def calculate_output_dims(args, kwargs):
        args, kwargs = ArgumentHelper.preprocess_args(args, kwargs)
        return get_dims(args[0])
    
    @staticmethod
    def calculate_output_ambiguous_dims(args, kwargs):
        args, kwargs = ArgumentHelper.preprocess_args(args, kwargs)
        return {dim for ambiguous_dims in get_ambiguous_dims(args[0]) for dim in ambiguous_dims}


class MultiArgumentElementwise:
    @staticmethod
    def validate_args(args, kwargs):
        assert isinstance(args, (list, tuple))
        assert isinstance(kwargs, dict)
        assert all(isinstance(arg, einexpr.einarray) for arg in args)
        assert all(not isinstance(arg, einexpr.einarray) for arg in kwargs.values())
    
    @staticmethod
    def process_args(args, kwargs):
        args, kwargs = ArgumentHelper.preprocess_args(args, kwargs)
        raw_aligned_arrays = einexpr.backends.align_arrays(*args)
        return raw_aligned_arrays, kwargs

    @staticmethod
    def calculate_output_dims(args, kwargs):
        args, kwargs = ArgumentHelper.preprocess_args(args, kwargs)
        return einexpr.dimension_utils.get_final_aligned_dims(*(get_dims(arg) for arg in args))
    
    @staticmethod
    def calculate_output_ambiguous_dims(args, kwargs):
        args, kwargs = ArgumentHelper.preprocess_args(args, kwargs)
        ambiguous_dims = einexpr.dimension_utils.calculate_ambiguous_final_aligned_dims(*(get_dims(arg) for arg in args))
        ambiguous_dims |= {dim for arg in args for dim in get_ambiguous_dims(arg)}
        return ambiguous_dims


class MultiDimensionReduction:
    @staticmethod
    def validate_args(args, kwargs):
        # TODO:
        # - Check that axes are in the array
        # - raise error when there are duplicate axes, esp of different types (Dimension, int, and negative int)
        assert isinstance(args, (list, tuple))
        assert isinstance(kwargs, dict)
        assert isinstance(args[0], einexpr.einarray)
        assert all(not isinstance(arg, einexpr.einarray) for arg in args[1:]) # TODO: mapreduce over pytree
        assert all(not isinstance(arg, einexpr.einarray) for arg in kwargs.values())
        assert 'axis' in args        
    
    @staticmethod
    def _calculate_axis(args, kwargs):
        axis = kwargs.get('axis')
        if axis is None:
            axis = list(range(len(get_dims(args[0]))))
        elif isinstance(axis, (Dimension, int)):
            axis = [axis]
        axis = [get_dims(args[0]).index(dim) if isinstance(dim, Dimension) else dim for dim in axis]
        assert all(isinstance(dim, int) for dim in axis)
        axis = [i if i >= 0 else i + len(get_dims(args[0])) for i in axis]
        return tuple(axis)
    
    @staticmethod
    def process_args(args, kwargs):
        args, kwargs = ArgumentHelper.preprocess_args(args, kwargs)
        axis = MultiDimensionReduction._calculate_axis(args, kwargs)
        return (get_raw(args[0]), *args[1:]), {**kwargs, 'axis': axis}

    @staticmethod
    def calculate_output_dims(args, kwargs):
        args, kwargs = ArgumentHelper.preprocess_args(args, kwargs)
        axis = MultiDimensionReduction._calculate_axis(args, kwargs)
        return [dim for i, dim in enumerate(get_dims(args[0])) if i not in axis]
    
    @staticmethod
    def calculate_output_ambiguous_dims(args, kwargs):
        args, kwargs = ArgumentHelper.preprocess_args(args, kwargs)
        axis = MultiDimensionReduction._calculate_axis(args, kwargs)
        ambiguous_out_dims = get_ambiguous_dims(args[0]) - {dim for i, dim in enumerate(get_dims(args[0])) if i in axis}
        return ambiguous_out_dims


class SingleDimensionReduction(MultiDimensionReduction):
    @staticmethod
    def validate_args(args, kwargs):
        args, kwargs = ArgumentHelper.preprocess_args(args, kwargs)
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
        assert isinstance(args[0], (list, tuple)) and all(isinstance(arg, einexpr.einarray) for arg in args[0])
        assert all(not isinstance(arg, einexpr.einarray) for arg in args[1:])
        if kwargs['axis'] is None:
            raise ValueError("Using ``axis=None`` (flattening concatenation) is not yet supported")
    
    @staticmethod
    def _calculate_axes(args, kwargs):
        args, kwargs = ArgumentHelper.preprocess_args(args, kwargs)
        axes = kwargs['axis']
        if isinstance(axes, (Dimension, int)):
            axes = [axes] * len(args[0])
        if not isinstance(axes, (list, tuple)):
            raise ValueError("Axes must be a list or tuple")
        axes = list(axes)
        for i, (axis, arg) in enumerate(zip(axes, args[0])):
            # If the axis is an integer, convert it to a Dimension
            if isinstance(axis, int):
                axis = get_dims(arg)[axis]
                axes[i] = axis
            elif not isinstance(axis, Dimension):
                raise ValueError(f"Invalid axis {axis}")
            assert axis not in get_ambiguous_dims(arg)
        return axes

    @staticmethod
    def process_args(args, kwargs):
        args, kwargs = ArgumentHelper.preprocess_args(args, kwargs)
        axes = Concatenation._calculate_axes(args, kwargs)
        axis_num = get_dims(args[0][0]).index(axes[0])
        # Align the arrays. Use the shape of the first array as the template.
        aligned_args = [args[0][0]]
        for axis, arg in zip(axes[1:], args[0][1:]):
            aligned_args.append(arg[tuple(dim if dim != axes[0] else axis for dim in get_dims(args[0][0]))])
        raw_aligned_args = [get_raw(arg) for arg in aligned_args]
        return (raw_aligned_args, *args[1:]), {**kwargs, 'axis': axis_num}

    @staticmethod
    def calculate_output_dims(args, kwargs):
        axes = Concatenation._calculate_axes(args, kwargs)
        # Check that arrays share all dimensions except the concatenated ones
        out_dims_set = set(get_dims(args[0][0])) - {axes[0]}
        for axis, arg in zip(axes[1:], args[0][1:]):
            arg_dims_set = set(get_dims(arg)) - {axis}
            if arg_dims_set != out_dims_set:
                raise ValueError(f"Arrays must have all the same dimensions except those concatentated over. The first input array {args[0][0]} has non-concatenated dimensions {out_dims_set}, while the input array {arg} has non-concatenated dimensions {arg_dims_set}.")
        out_dims = [dim if dim != axes[0] else tuple(axes) for dim in get_dims(args[0][0])]
        return out_dims
        
    @staticmethod
    def calculate_output_ambiguous_dims(args, kwargs):
        args, kwargs = ArgumentHelper.preprocess_args(args, kwargs)
        axes = Concatenation._calculate_axes(args, kwargs)
        ambiguous_dims = {dim for arg in args[0] for dim in get_ambiguous_dims(arg) if dim not in axes}
        for axis, arg in zip(axes[1:], args[0][1:]):
            for dim0, dim in zip(get_dims(args[0][0]), get_dims(arg)):
                if axes[0] != dim0 and dim != axis and dim0 != dim:
                    ambiguous_dims.add(dim)
        return ambiguous_dims
