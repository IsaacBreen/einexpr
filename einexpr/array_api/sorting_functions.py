from ._types import array
import einexpr


def argsort(x: array, /, *, axis: int = -1, descending: bool = False, stable: bool = True) -> array:
    """
    Returns the indices that sort an array ``x`` along a specified axis.

    Parameters
    ----------
    x : array
        input array.
    axis: int
        axis along which to sort. If set to ``-1``, the function must sort along the last axis. Default: ``-1``.
    descending: bool
        sort order. If ``True``, the returned indices sort ``x`` in descending order (by value). If ``False``, the returned indices sort ``x`` in ascending order (by value). Default: ``False``.
    stable: bool
        sort stability. If ``True``, the returned indices must maintain the relative order of ``x`` values which compare as equal. If ``False``, the returned indices may or may not maintain the relative order of ``x`` values which compare as equal (i.e., the relative order of ``x`` values which compare as equal is implementation-dependent). Default: ``True``.

    Returns
    -------
    out : array
        an array of indices. The returned array must have the same shape as ``x``. The returned array must have the default array index data type.
    """
    args = (x,)
    kwargs = {'axis': axis, 'descending': descending, 'stable': stable}
    helper = einexpr.dimension_utils.MultipleArgumentElementwise
    helper.validate_args(args, kwargs)
    out_dims = helper.calculate_output_dims(args, kwargs)
    ambiguous_dims = helper.calculate_output_ambiguous_dims(args, kwargs)
    processed_args, processed_kwargs = helper.process_args(args, kwargs)
    result = einexpr.einarray(
        x.a.__array_namespace__().argsort(*processed_args, **processed_kwargs), 
        dims=out_dims, 
        ambiguous_dims=ambiguous_dims)
    return result

def sort(x: array, /, *, axis: int = -1, descending: bool = False, stable: bool = True) -> array:
    """
    Returns a sorted copy of an input array ``x``.

    Parameters
    ----------
    x: array
        input array.
    axis: int
        axis along which to sort. If set to ``-1``, the function must sort along the last axis. Default: ``-1``.
    descending: bool
        sort order. If ``True``, the array must be sorted in descending order (by value). If ``False``, the array must be sorted in ascending order (by value). Default: ``False``.
    stable: bool
        sort stability. If ``True``, the returned array must maintain the relative order of ``x`` values which compare as equal. If ``False``, the returned array may or may not maintain the relative order of ``x`` values which compare as equal (i.e., the relative order of ``x`` values which compare as equal is implementation-dependent). Default: ``True``.

    Returns
    -------
    out : array
        a sorted array. The returned array must have the same data type and shape as ``x``.
    """
    args = (x,)
    kwargs = {'axis': axis, 'descending': descending, 'stable': stable}
    helper = einexpr.dimension_utils.MultipleArgumentElementwise
    helper.validate_args(args, kwargs)
    out_dims = helper.calculate_output_dims(args, kwargs)
    ambiguous_dims = helper.calculate_output_ambiguous_dims(args, kwargs)
    processed_args, processed_kwargs = helper.process_args(args, kwargs)
    result = einexpr.einarray(
        x.a.__array_namespace__().sort(*processed_args, **processed_kwargs), 
        dims=out_dims, 
        ambiguous_dims=ambiguous_dims)
    return result

__all__ = ['argsort', 'sort']