from click import Argument
import einexpr
from regex import R


backend_array_types = dict()
backend_dim_kwargs_to_resolve = ['dim', 'axis']

function_registry = []


def is_dimensionless(array):
    return isinstance(array, (int, float)) or einexpr.backends.conforms_to_array_api(array) and len(array.shape) == 0


# TODO: can safely remove the following three methods now that proprocessing is done in the einarray magic calls
def get_dims(array):
    if isinstance(array, einexpr.array):
        return array.dims
    elif is_dimensionless(array):
        return ()
    else:
        raise TypeError(f'{array} is not a recognized einexpr array')


def get_ambiguous_dims(array):
    if isinstance(array, einexpr.array):
        return array.ambiguous_dims
    elif is_dimensionless(array):
        return ()
    else:
        raise TypeError(f'{array} is not a recognized einexpr array')


def get_raw(array):
    if isinstance(array, einexpr.array):
        return array.a
    elif is_dimensionless(array):
        return array
    else:
        raise TypeError(f'{array} is not a einexpr.backends.conforms_to_array_api einexpr array')


class ArgumentHelper:
    @staticmethod
    def preprocess_arg(arg):
            if isinstance(arg, einexpr.array):
                return arg
            elif is_dimensionless(arg):
                return einexpr.array(arg, dims=())
            else:
                return arg
                
    @staticmethod
    def preprocess_args(args, kwargs):
        args = [ArgumentHelper.preprocess_arg(arg) for arg in args]
        # kwargs = {key: ArgumentHelper.preprocess_arg(arg) for key, arg in kwargs.items()}
        return args, kwargs

    def __init__(self, args, kwargs):
        self.args, self.kwargs = ArgumentHelper.preprocess_args(args, kwargs)


class MultiArgumentElementwise:
    @staticmethod
    def validate_args(args, kwargs):
        assert isinstance(args, (list, tuple))
        assert isinstance(kwargs, dict)
        # assert all(isinstance(arg, (einexpr.einarray, int, float)) for arg in args)
    
    @staticmethod
    def process_args(args, kwargs):
        # args, kwargs = ArgumentHelper.preprocess_args(args, kwargs)
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


SingleArgumentElementwise = MultiArgumentElementwise

class MultiDimensionReduction:
    @staticmethod
    def validate_args(args, kwargs):
        # TODO:
        # - Check that axes are in the array
        # - raise error when there are duplicate axes, esp of different types (einexpr.types.Dimension, int, and negative int)
        assert isinstance(args, (list, tuple))
        assert isinstance(kwargs, dict)
        assert all(isinstance(arg, (einexpr.einarray, int, float)) for arg in args)
        assert 'axis' in args
    
    @staticmethod
    def _calculate_axis(args, kwargs):
        axis = kwargs.get('axis')
        if axis is None:
            axis = list(range(len(get_dims(args[0]))))
        elif isinstance(axis, (einexpr.types.Dimension, int)):
            axis = [axis]
        axis = [get_dims(args[0]).index(dim) if isinstance(dim, einexpr.types.Dimension) else dim for dim in axis]
        assert all(isinstance(dim, int) for dim in axis)
        axis = [i if i >= 0 else i + len(get_dims(args[0])) for i in axis]
        return tuple(axis)
    
    @staticmethod
    def process_args(args, kwargs):
        # args, kwargs = ArgumentHelper.preprocess_args(args, kwargs)
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
        assert 'axis' in kwargs
        assert kwargs['axis'] is None or isinstance(kwargs['axis'], (einexpr.types.Dimension, int))
        MultiDimensionReduction.validate_args(args, kwargs)


class Concatenation:
    @staticmethod
    def validate_args(args, kwargs):
        assert isinstance(args, (list, tuple))
        assert isinstance(kwargs, dict)
        assert isinstance(args[0], (list, tuple)) and all(isinstance(arg, einexpr.einarray) for arg in args[0])
        if kwargs['axis'] is None:
            raise ValueError("Using ``axis=None`` (flattening concatenation) is not yet supported")
    
    @staticmethod
    def _calculate_axes(args, kwargs):
        args, kwargs = ArgumentHelper.preprocess_args(args, kwargs)
        axes = kwargs['axis']
        if isinstance(axes, (einexpr.types.Dimension, int)):
            axes = [axes] * len(args[0])
        if not isinstance(axes, (list, tuple)):
            raise ValueError("Axes must be a list or tuple")
        axes = list(axes)
        for i, (axis, arg) in enumerate(zip(axes, args[0])):
            # If the axis is an integer, convert it to a einexpr.types.Dimension
            if isinstance(axis, int):
                axis = get_dims(arg)[axis]
                axes[i] = axis
            elif not isinstance(axis, einexpr.types.Dimension):
                raise ValueError(f"Invalid axis {axis}")
            assert axis not in get_ambiguous_dims(arg)
        return axes

    @staticmethod
    def process_args(args, kwargs):
        # args, kwargs = ArgumentHelper.preprocess_args(args, kwargs)
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


__all__ = [
    'get_dims',
    'get_ambiguous_dims',
    'get_raw',
    'is_dimensionless',
    'SingleArgumentElementwise', 
    'MultiArgumentElementwise', 
    'SingleDimensionReduction', 
    'MultiDimensionReduction',
    'Concatenation'
]