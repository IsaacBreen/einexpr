from ._types import Union, array, dtype, finfo_object, iinfo_object
import einexpr


def astype(x: array, dtype: dtype, /, *, copy: bool = True) -> array:
    """
    Copies an array to a specified data type irrespective of :ref:`type-promotion` rules.

    .. note::
       Casting floating-point ``NaN`` and ``infinity`` values to integral data types is not specified and is implementation-dependent.

    .. note::
       When casting a boolean input array to a real-valued data type, a value of ``True`` must cast to a numeric value equal to ``1``, and a value of ``False`` must cast to a numeric value equal to ``0``.

       When casting a real-valued input array to ``bool``, a value of ``0`` must cast to ``False``, and a non-zero value must cast to ``True``.

    Parameters
    ----------
    x: array
        array to cast.
    dtype: dtype
        desired data type.
    copy: bool
        specifies whether to copy an array when the specified ``dtype`` matches the data type of the input array ``x``. If ``True``, a newly allocated array must always be returned. If ``False`` and the specified ``dtype`` matches the data type of the input array, the input array must be returned; otherwise, a newly allocated must be returned. Default: ``True``.

    Returns
    -------
    out: array
        an array having the specified data type. The returned array must have the same shape as ``x``.
    """
    return einexpr.einarray(
        x.a.__array_namespace__().astype(x.a, dtype, copy=copy),
        dims=x.dims,
        ambiguous_dims=x.ambiguous_dims)

def can_cast(from_: Union[dtype, array], to: dtype, /) -> bool:
    """
    Determines if one data type can be cast to another data type according :ref:`type-promotion` rules.

    Parameters
    ----------
    from_: Union[dtype, array]
        input data type or array from which to cast.
    to: dtype
        desired data type.

    Returns
    -------
    out: bool
        ``True`` if the cast can occur according to :ref:`type-promotion` rules; otherwise, ``False``.
    """
    if isinstance(from_, einexpr.einarray):
        return from_.a.__array_namespace__().can_cast(from_.a, to)
    else:
        raise TypeError("can_cast() only accepts einarray objects. Use your backend's can_cast() function instead.")

def finfo(type: Union[dtype, array], /) -> finfo_object:
    """
    Machine limits for real-valued floating-point data types.

    Parameters
    ----------
    type: Union[dtype, array]
        the kind of real-valued floating-point data-type about which to get information.

    Returns
    -------
    out: finfo object
        an object having the following attributes:

        - **bits**: *int*

          number of bits occupied by the floating-point data type.

        - **eps**: *float*

          difference between 1.0 and the next smallest representable floating-point number larger than 1.0 according to the IEEE-754 standard.

        - **max**: *float*

          largest representable number.

        - **min**: *float*

          smallest representable number.

        - **smallest_normal**: *float*

          smallest positive floating-point number with full precision.
    """
    raise NotImplementedError("Use your backend's finfo() function instead.")

def iinfo(type: Union[dtype, array], /) -> iinfo_object:
    """
    Machine limits for integer data types.

    Parameters
    ----------
    type: Union[dtype, array]
        the kind of integer data-type about which to get information.

    Returns
    -------
    out: iinfo object
        an object having the following attributes:

        - **bits**: *int*

          number of bits occupied by the type.

        - **max**: *int*

          largest representable number.

        - **min**: *int*

          smallest representable number.
    """
    raise NotImplementedError("Use your backend's iinfo() function instead.")

def result_type(*arrays_and_dtypes: Union[array, dtype]) -> dtype:
    """
    Returns the dtype that results from applying the type promotion rules (see :ref:`type-promotion`) to the arguments.

    .. note::
       If provided mixed dtypes (e.g., integer and floating-point), the returned dtype will be implementation-specific.

    Parameters
    ----------
    arrays_and_dtypes: Union[array, dtype]
        an arbitrary number of input arrays and/or dtypes.

    Returns
    -------
    out: dtype
        the dtype resulting from an operation involving the input arrays and dtypes.
    """
    # Use the array namspace first array API-conformant array.
    for x in arrays_and_dtypes:
        if einexpr.backends.conforms_to_array_api(x):
            return x.__array_namespace__().result_type(*arrays_and_dtypes)
    # If all arrays are dtypes, raise an error.
    raise TypeError("result_type() only accepts einarray objects. Use your backend's result_type() function instead.")

__all__ = ['astype', 'can_cast', 'finfo', 'iinfo', 'result_type']
