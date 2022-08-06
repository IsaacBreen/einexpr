from typing import Iterable, Tuple
from click import Argument
import einexpr


backend_array_types = dict()
backend_dim_kwargs_to_resolve = ['dim', 'axis']

function_registry = []


def is_dimensionless(array):
    return isinstance(array, (int, float)) or einexpr.backends.conforms_to_array_api(array) and len(array.shape) == 0


# TODO: can safely remove the following three methods now that proprocessing is done in the einarray magic calls
def get_dims(array):
    if isinstance(array, einexpr.array):
        return array.dims.dimensions
    elif is_dimensionless(array):
        return ()
    else:
        raise TypeError(f'{array} is not a recognized einexpr array')


def get_dimspec(array):
    if isinstance(array, einexpr.array):
        return array.dims
    elif is_dimensionless(array):
        return einexpr.array_api.dimension.DimensionSpecification((), {})
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


def resolve_absorbable_dims(arrays: Tuple[einexpr.array_api.array, ...]) -> Tuple[einexpr.array_api.array, ...]:
    resolved_arg_dimensions = einexpr.dimension_utils.resolve_positional_dims(tuple((..., *get_dims(array)) for array in arrays))
    resolved_arg_dimensions = [dimensions[1:] for dimensions in resolved_arg_dimensions]
    resolved_arg_dimspecs = tuple(get_dimspec(array).with_dimensions(dimensions) for array, dimensions in zip(arrays, resolved_arg_dimensions))
    return tuple(
        einexpr.einarray(get_raw(arg), dims=dimspec)
        if isinstance(arg, einexpr.array) else arg
        for arg, dimspec in zip(arrays, resolved_arg_dimspecs)
    )



class ArgumentHelper:
    @staticmethod
    def prepreprocess_arg(arg):
        if isinstance(arg, einexpr.array):
            return arg
        elif is_dimensionless(arg):
            return einexpr.array(arg, dims=())
        elif einexpr.backends.conforms_to_array_api(arg):
            return einexpr.array(arg)
        else:
            return arg
            
    @staticmethod
    def prepreprocess_args(args, kwargs):
        args = tuple(ArgumentHelper.prepreprocess_arg(arg) for arg in args)
        # kwargs = {key: ArgumentHelper.prepreprocess_arg(arg) for key, arg in kwargs.items()}
        return einexpr.backends.tracer.apply_tracer((args, kwargs))

    @staticmethod
    def preprocess_arg(arg):
        if isinstance(arg, einexpr.array):
            return arg
        elif is_dimensionless(arg):
            return arg
        elif einexpr.backends.conforms_to_array_api(arg):
            return einexpr.array(arg)
        else:
            return arg
            
    @staticmethod
    def preprocess_args(args, kwargs):
        args = tuple(ArgumentHelper.preprocess_arg(arg) for arg in args)
        # kwargs = {key: ArgumentHelper.preprocess_arg(arg) for key, arg in kwargs.items()}
        return args, kwargs

    def __init__(self, args, kwargs):
        self.args, self.kwargs = ArgumentHelper.prepreprocess_args(args, kwargs)


class MultipleArgumentElementwise:
    @staticmethod
    def validate_args(args, kwargs):
        assert isinstance(args, (list, tuple))
        assert isinstance(kwargs, dict)
        assert all(is_dimensionless(arg) or einexpr.backends.conforms_to_array_api(arg) for arg in args)
    
    @staticmethod
    def process_args(args, kwargs):
        args, kwargs = ArgumentHelper.preprocess_args(args, kwargs)
        args = resolve_absorbable_dims(args)
        raw_aligned_arrays = einexpr.backends.align_arrays(*args)
        return raw_aligned_arrays, kwargs

    @staticmethod
    def calculate_output_dims(args, kwargs):
        args, kwargs = ArgumentHelper.prepreprocess_args(args, kwargs)
        args = resolve_absorbable_dims(args)
        return einexpr.dimension_utils.get_final_aligned_dims(*(get_dims(arg) for arg in args))
    
    @staticmethod
    def calculate_output_ambiguous_dims(args, kwargs):
        args, kwargs = ArgumentHelper.prepreprocess_args(args, kwargs)
        args = resolve_absorbable_dims(args)
        ambiguous_dims = einexpr.dimension_utils.calculate_ambiguous_final_aligned_dims(*(get_dims(arg) for arg in args))
        ambiguous_dims |= {dim for arg in args for dim in get_ambiguous_dims(arg)}
        return ambiguous_dims


SingleArgumentElementwise = MultipleArgumentElementwise


class SingleArgumentMultipleDimensionReduction:
    @staticmethod
    def validate_args(args, kwargs):
        # TODO:
        # - Check that axes are in the array
        # - raise error when there are duplicate axes, esp of different types (einexpr.array_api.dimension.AtomicDimension, int, and negative int)
        assert isinstance(args, (list, tuple))
        assert isinstance(kwargs, dict)
        assert len(args) == 1
        assert is_dimensionless(args[0]) or einexpr.backends.conforms_to_array_api(args[0])
        assert 'axis' in kwargs
    
    @staticmethod
    def _calculate_named_axis(args, kwargs):
        array = args[0]
        axis = kwargs.get('axis')
        if axis is None:
            axis = tuple(range(len(get_dims(args[0]))))
        elif isinstance(axis, (einexpr.array_api.dimension.AtomicDimension, int)):
            axis = (axis,)
        elif isinstance(axis, set):
            axis = tuple(axis)
        _axis = []
        array_dims = get_dims(array)
        # Convert all integer axes into AtomicDimensions
        for i, dim in enumerate(axis):
            if isinstance(dim, (list, tuple, einexpr.array_api.dimension.AtomicDimension, einexpr.array_api.dimension.AbsorbingDimension)):
                _axis.append(dim)
            elif isinstance(dim, int):
                _axis.append(array_dims[dim])
            else:
                raise TypeError(f'{dim} is not a recognized dimension')
        axis = tuple(_axis)
        # Ensure all dimensions are unique
        flattened_axis = einexpr.dimension_utils.expand_dims(axis)
        if len(flattened_axis) != len(set(flattened_axis)):
            raise ValueError(f'Duplicate dimensions in axis: {axis}')
        # # Isolate the dimensions to be reduced
        # array_dims = einexpr.dimension_utils.isolate_dims(array_dims, axis)
        return axis
    
    @staticmethod
    def _calculate_integer_axis(args, kwargs):
        array: einexpr.einarray = args[0]
        named_axis = SingleArgumentMultipleDimensionReduction._calculate_named_axis(args, kwargs)
        array_dims = einexpr.dimension_utils.isolate_dims(get_dims(array), named_axis)
        axis = tuple(array_dims.index(dim) for dim in named_axis)
        return axis
    
    @staticmethod
    def process_args(args, kwargs):
        args, kwargs = ArgumentHelper.preprocess_args(args, kwargs)
        named_axis = SingleArgumentMultipleDimensionReduction._calculate_named_axis(args, kwargs)
        integer_axis = SingleArgumentMultipleDimensionReduction._calculate_integer_axis(args, kwargs)
        array: einexpr.einarray = args[0].isolate_dims(named_axis)
        return (get_raw(array), *args[1:]), {**kwargs, 'axis': integer_axis}

    @staticmethod
    def calculate_output_dims(args, kwargs):
        args, kwargs = ArgumentHelper.prepreprocess_args(args, kwargs)
        # Convert all integer axes into dimension objects
        named_axis = SingleArgumentMultipleDimensionReduction._calculate_named_axis(args, kwargs)
        array: einexpr.einarray = args[0]
        array_dims = einexpr.dimension_utils.isolate_dims(get_dims(array), named_axis)
        assert set(array_dims) & set(named_axis) == set(named_axis), 'All dimensions in axis must be in the array'
        output_dims = tuple(dim for dim in array_dims if dim not in named_axis)
        output_dims = einexpr.dimension_utils.parse_dims(output_dims)
        sizes = einexpr.dimension_utils.gather_sizes(get_dimspec(array))
        sizes = {dim: size for dim, size in sizes.items() if einexpr.utils.tree_contains(output_dims, dim)}
        dimspec = einexpr.array_api.dimension.DimensionSpecification(output_dims, sizes)
        return dimspec
    
    @staticmethod
    def calculate_output_ambiguous_dims(args, kwargs):
        args, kwargs = ArgumentHelper.prepreprocess_args(args, kwargs)
        axis = SingleArgumentMultipleDimensionReduction._calculate_integer_axis(args, kwargs)
        ambiguous_out_dims = get_ambiguous_dims(args[0]) - {dim for i, dim in enumerate(get_dims(args[0])) if i in axis}
        return ambiguous_out_dims


class SingleArgumentSingleDimensionReduction(SingleArgumentMultipleDimensionReduction):
    @staticmethod
    def validate_args(args, kwargs):
        assert 'axis' in kwargs
        assert kwargs['axis'] is None or isinstance(kwargs['axis'], (int, list, tuple, einexpr.array_api.dimension.AtomicDimension))
        SingleArgumentMultipleDimensionReduction.validate_args(args, kwargs)


class Concatenation:
    @staticmethod
    def validate_args(args, kwargs):
        assert isinstance(args, (list, tuple))
        assert isinstance(kwargs, dict)
        assert len(args) == 1
        assert isinstance(args[0], (list, tuple))
        assert all(is_dimensionless(arg) or einexpr.backends.conforms_to_array_api(arg) for arg in args[0])
        if kwargs['axis'] is None:
            raise ValueError("Using ``axis=None`` (flattening concatenation) is not yet supported")
    
    @staticmethod
    def _calculate_axes(args, kwargs):
        args, kwargs = ArgumentHelper.prepreprocess_args(args, kwargs)
        axes = kwargs['axis']
        if isinstance(axes, (einexpr.array_api.dimension.AtomicDimension, int)):
            axes = [axes] * len(args[0])
        if not isinstance(axes, (list, tuple)):
            raise ValueError("Axes must be a list or tuple")
        axes = list(axes)
        for i, (axis, arg) in enumerate(zip(axes, args[0])):
            # If the axis is an integer, convert it to a einexpr.array_api.dimension.AtomicDimension
            if isinstance(axis, int):
                axis = get_dims(arg)[axis]
                axes[i] = axis
            elif not isinstance(axis, einexpr.array_api.dimension.AtomicDimension):
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
        args, kwargs = ArgumentHelper.prepreprocess_args(args, kwargs)
        axes = Concatenation._calculate_axes(args, kwargs)
        ambiguous_dims = {dim for arg in args[0] for dim in get_ambiguous_dims(arg) if dim not in axes}
        for axis, arg in zip(axes[1:], args[0][1:]):
            for dim0, dim in zip(get_dims(args[0][0]), get_dims(arg)):
                if axes[0] != dim0 and dim != axis and dim0 != dim:
                    ambiguous_dims.add(dim)
        return ambiguous_dims


__all__ = [
    'get_dims',
    'get_dimspec',
    'get_ambiguous_dims',
    'get_raw',
    'is_dimensionless',
    'SingleArgumentElementwise', 
    'MultipleArgumentElementwise', 
    'SingleArgumentSingleDimensionReduction', 
    'SingleArgumentMultipleDimensionReduction',
    'Concatenation'
]