from ._types import Optional, Tuple, array
import einexpr


def argmax(x: array, /, *, axis: Optional[int] = None, keepdims: bool = False) -> array:
    """
    Returns the indices of the maximum values along a specified axis. When the maximum value occurs multiple times, only the indices corresponding to the first occurrence are returned.

    Parameters
    ----------
    x: array
        input array. Should have a real-valued data type.
    axis: Optional[int]
        axis along which to search. If ``None``, the function must return the index of the maximum value of the flattened array. Default: ``None``.
    keepdims: bool
        if ``True``, the reduced axes (dimensions) must be included in the result as singleton dimensions, and, accordingly, the result must be compatible with the input array (see :ref:`broadcasting`). Otherwise, if ``False``, the reduced axes (dimensions) must not be included in the result. Default: ``False``.

    Returns
    -------
    out: array
        if ``axis`` is ``None``, a zero-dimensional array containing the index of the first occurrence of the maximum value; otherwise, a non-zero-dimensional array containing the indices of the maximum values. The returned array must have be the default array index data type.
    """
    args = (x,)
    kwargs = {'axis': axis, 'keepdims': keepdims}
    helper = einexpr.dimension_utils.SingleArgumentMultipleDimensionReduction
    helper.validate_args(args, kwargs)
    out_dims = helper.calculate_output_dims(args, kwargs)
    ambiguous_dims = helper.calculate_output_ambiguous_dims(args, kwargs)
    processed_args, processed_kwargs = helper.process_args(args, kwargs)
    result = einexpr.einarray(
        einexpr.backends.get_array_api_backend(array=x.a).argmax(*processed_args, **processed_kwargs), 
        dims=out_dims, 
        ambiguous_dims=ambiguous_dims)
    return result

def argmin(x: array, /, *, axis: Optional[int] = None, keepdims: bool = False) -> array:
    """
    Returns the indices of the minimum values along a specified axis. When the minimum value occurs multiple times, only the indices corresponding to the first occurrence are returned.

    Parameters
    ----------
    x: array
        input array. Should have a real-valued data type.
    axis: Optional[int]
        axis along which to search. If ``None``, the function must return the index of the minimum value of the flattened array. Default: ``None``.
    keepdims: bool
        if ``True``, the reduced axes (dimensions) must be included in the result as singleton dimensions, and, accordingly, the result must be compatible with the input array (see :ref:`broadcasting`). Otherwise, if ``False``, the reduced axes (dimensions) must not be included in the result. Default: ``False``.

    Returns
    -------
    out: array
        if ``axis`` is ``None``, a zero-dimensional array containing the index of the first occurrence of the minimum value; otherwise, a non-zero-dimensional array containing the indices of the minimum values. The returned array must have the default array index data type.
    """
    args = (x,)
    kwargs = {'axis': axis, 'keepdims': keepdims}
    helper = einexpr.dimension_utils.SingleArgumentMultipleDimensionReduction
    helper.validate_args(args, kwargs)
    out_dims = helper.calculate_output_dims(args, kwargs)
    ambiguous_dims = helper.calculate_output_ambiguous_dims(args, kwargs)
    processed_args, processed_kwargs = helper.process_args(args, kwargs)
    result = einexpr.einarray(
        einexpr.backends.get_array_api_backend(array=x.a).argmin(*processed_args, **processed_kwargs), 
        dims=out_dims, 
        ambiguous_dims=ambiguous_dims)
    return result

def nonzero(x: array, /) -> Tuple[array, ...]:
    """
    Returns the indices of the array elements which are non-zero.

    .. note::
       If ``x`` has a complex floating-point data type, non-zero elements are those elements having at least one component (real or imaginary) which is non-zero.

    .. note::
       If ``x`` has a boolean data type, non-zero elements are those elements which are equal to ``True``.

    .. admonition:: Data-dependent output shape
       :class: admonition important

       The shape of the output array for this function depends on the data values in the input array; hence, array libraries which build computation graphs (e.g., JAX, Dask, etc.) may find this function difficult to implement without knowing array values. Accordingly, such libraries may choose to omit this function. See :ref:`data-dependent-output-shapes` section for more details.

    Parameters
    ----------
    x: array
        input array. Must have a positive rank. If ``x`` is zero-dimensional, the function must raise an exception.

    Returns
    -------
    out: Typle[array, ...]
        a tuple of ``k`` arrays, one for each dimension of ``x`` and each of size ``n`` (where ``n`` is the total number of non-zero elements), containing the indices of the non-zero elements in that dimension. The indices must be returned in row-major, C-style order. The returned array must have the default array index data type.
    """
    raise NotImplementedError

def where(condition: array, x1: array, x2: array, /) -> array:
    """
    Returns elements chosen from ``x1`` or ``x2`` depending on ``condition``.

    Parameters
    ----------
    condition: array
        when ``True``, yield ``x1_i``; otherwise, yield ``x2_i``. Must be compatible with ``x1`` and ``x2`` (see :ref:`broadcasting`).
    x1: array
        first input array. Must be compatible with ``condition`` and ``x2`` (see :ref:`broadcasting`).
    x2: array
        second input array. Must be compatible with ``condition`` and ``x1`` (see :ref:`broadcasting`).

    Returns
    -------
    out: array
        an array with elements from ``x1`` where ``condition`` is ``True``, and elements from ``x2`` elsewhere. The returned array must have a data type determined by :ref:`type-promotion` rules with the arrays ``x1`` and ``x2``.
    """
    args = (condition, x1, x2,)
    kwargs = {}
    helper = einexpr.dimension_utils.MultipleArgumentElementwise
    helper.validate_args(args, kwargs)
    out_dims = helper.calculate_output_dims(args, kwargs)
    ambiguous_dims = helper.calculate_output_ambiguous_dims(args, kwargs)
    processed_args, processed_kwargs = helper.process_args(args, kwargs)
    result = einexpr.einarray(
        einexpr.backends.get_array_api_backend(array=x2.a).where(*processed_args, **processed_kwargs), 
        dims=out_dims, 
        ambiguous_dims=ambiguous_dims)
    return result

__all__ = ['argmax', 'argmin', 'nonzero', 'where']
