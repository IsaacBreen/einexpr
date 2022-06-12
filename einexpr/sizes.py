import numpy as np
import numpy.typing as npt
from numpy.typing import ArrayLike, DTypeLike, _ShapeLike
import typing
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union


size_calcs = {}

def register_out_shape(*shaped_funcs):
    """
    Register a function that precalculates shape of the output of a function.
    """
    def decorator(shape_precalculator):
        for shaped_func in shaped_funcs:
            size_calcs[shaped_func] = shape_precalculator
        return shape_precalculator
    return decorator


def calculate_output_shape(func, *args, **kwargs):
    """
    Calculate the shape of the output of a function.
    """
    if func in size_calcs:
        return size_calcs[func](*args, **kwargs)
    else:
        raise ValueError("No shape precalculator for function {}".format(func))


class NewIndex:
    pass


@register_out_shape(np.sum)
def numpy_sum_output_dims(a: ArrayLike, axis: Optional[Union[int, Sequence[int]]] = None, dtype: Optional[DTypeLike] = None, out: Optional[ArrayLike] = None, keepdims: Optional[bool] = None, initial: Optional[Any] = None, where: Optional[ArrayLike] = None) -> ArrayLike:
    if axis is None:
        return ()
    if isinstance(axis, int):
        return tuple(i for i in range(a.ndim) if i != axis)
    else:
        return tuple(i for i in range(a.ndim) if i not in axis)


@register_out_shape(np.concatenate)
def numpy_concatenate_output_dims(a: Sequence[ArrayLike], axis: Optional[int] = None, out: Optional[ArrayLike] = None, dtype: Optional[DTypeLike] = None, casting: Optional[str] = None) -> ArrayLike:
    if axis is None:
        return (NewIndex(),)
    else:
        return tuple(i if i != axis else NewIndex() for i in range(a[0].ndim))

