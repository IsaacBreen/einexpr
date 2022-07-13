from ._types import array
from .. import SingleArgumentElementwise, einarray
from .. import einarray
from .. import SingleArgumentElementwise, einarray

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
    out_dims = SingleArgumentElementwise.calculate_output_dims(args, kwargs)
    ambiguous_dims = SingleArgumentElementwise.calculate_output_ambiguous_dims(args, kwargs)
    processed_args, processed_kwargs = SingleArgumentElementwise.process_args(args, kwargs)
    result = einarray(argsort(*processed_args, **processed_kwargs), dims=out_dims, ambiguous_dims=ambiguous_dims)
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
    out_dims = {implementation_helper_name}.calculate_output_dims({params})
    ambiguous_dims = {implementation_helper_name}.calculate_output_ambiguous_dims(args, kwargs)
    processed_args, processed_kwargs = {implementation_helper_name}.process_args(args, kwargs)
    result = einarray({func_name}(*processed_args, **processed_kwargs), dims=out_dims, ambiguous_dims=ambiguous_dims)
    return result

__all__ = ['argsort', 'sort']