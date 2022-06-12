"""
Calculates output dimensions of numpy functions, ignoring trivial expanded dimensions.
"""
import numpy as np
import numpy.typing as npt
from numpy.typing import ArrayLike, DTypeLike, _ShapeLike
import typing
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union


class NewIndex:
    pass

"""
Sum of array elements over a given axis.

Parameters
----------
a : array_like
    Elements to sum.
axis : None or int or tuple of ints, optional
    Axis or axes along which a sum is performed.  The default,
    axis=None, will sum all of the elements of the input array.  If
    axis is negative it counts from the last to the first axis.

    .. versionadded:: 1.7.0

    If axis is a tuple of ints, a sum is performed on all of the axes
    specified in the tuple instead of a single axis or all the axes as
    before.
dtype : dtype, optional
    The type of the returned array and of the accumulator in which the
    elements are summed.  The dtype of `a` is used by default unless `a`
    has an integer dtype of less precision than the default platform
    integer.  In that case, if `a` is signed then the platform integer
    is used while if `a` is unsigned then an unsigned integer of the
    same precision as the platform integer is used.
out : ndarray, optional
    Alternative output array in which to place the result. It must have
    the same shape as the expected output, but the type of the output
    values will be cast if necessary.
keepdims : bool, optional
    If this is set to True, the axes which are reduced are left
    in the result as dimensions with size one. With this option,
    the result will broadcast correctly against the input array.

    If the default value is passed, then `keepdims` will not be
    passed through to the `sum` method of sub-classes of
    `ndarray`, however any non-default value will be.  If the
    sub-class' method does not implement `keepdims` any
    exceptions will be raised.
initial : scalar, optional
    Starting value for the sum. See `~numpy.ufunc.reduce` for details.

    .. versionadded:: 1.15.0

where : array_like of bool, optional
    Elements to include in the sum. See `~numpy.ufunc.reduce` for details.

    .. versionadded:: 1.17.0

Returns
-------
sum_along_axis : ndarray
    An array with the same shape as `a`, with the specified
    axis removed.   If `a` is a 0-d array, or if `axis` is None, a scalar
    is returned.  If an output array is specified, a reference to
    `out` is returned.
"""
def numpy_sum_output_dims(a: ArrayLike, axis: Optional[Union[int, Sequence[int]]] = None, dtype: Optional[DTypeLike] = None, out: Optional[ArrayLike] = None, keepdims: Optional[bool] = None, initial: Optional[Any] = None, where: Optional[ArrayLike] = None) -> ArrayLike:
    if axis is None:
        return ()
    if isinstance(axis, int):
        return tuple(i for i in range(a.ndim) if i != axis)
    else:
        return tuple(i for i in range(a.ndim) if i not in axis)


"""
Expand the shape of an array.

Insert a new axis that will appear at the `axis` position in the expanded
array shape.

Parameters
----------
a : array_like
    Input array.
axis : int or tuple of ints
    Position in the expanded axes where the new axis (or axes) is placed.

    .. deprecated:: 1.13.0
        Passing an axis where ``axis > a.ndim`` will be treated as
        ``axis == a.ndim``, and passing ``axis < -a.ndim - 1`` will
        be treated as ``axis == 0``. This behavior is deprecated.

    .. versionchanged:: 1.18.0
        A tuple of axes is now supported.  Out of range axes as
        described above are now forbidden and raise an `AxisError`.

Returns
-------
result : ndarray
    View of `a` with the number of dimensions increased.
"""
def numpy_expand_dims_output_dims(a: ArrayLike, axis: Union[int, Sequence[int]]) -> ArrayLike:
    return tuple(range(a.ndim))
    

"""
concatenate((a1, a2, ...), axis=0, out=None, dtype=None, casting="same_kind")

Join a sequence of arrays along an existing axis.

Parameters
----------
a1, a2, ... : sequence of array_like
    The arrays must have the same shape, except in the dimension
    corresponding to `axis` (the first, by default).
axis : int, optional
    The axis along which the arrays will be joined.  If axis is None,
    arrays are flattened before use.  Default is 0.
out : ndarray, optional
    If provided, the destination to place the result. The shape must be
    correct, matching that of what concatenate would have returned if no
    out argument were specified.
dtype : str or dtype
    If provided, the destination array will have this dtype. Cannot be
    provided together with `out`.

    .. versionadded:: 1.20.0

casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
    Controls what kind of data casting may occur. Defaults to 'same_kind'.

    .. versionadded:: 1.20.0

Returns
-------
res : ndarray
    The concatenated array.
"""
def numpy_concatenate_output_dims(a: Sequence[ArrayLike], axis: Optional[int] = None, out: Optional[ArrayLike] = None, dtype: Optional[DTypeLike] = None, casting: Optional[str] = None) -> ArrayLike:
    if axis is None:
        return (NewIndex(),)
    else:
        return tuple(i for i in range(a[0].ndim) if i != axis)


