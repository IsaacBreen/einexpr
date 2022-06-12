import numpy as np

size_calcs = {}

def register_out_shape(shaped_func):
    """
    Register a function that precalculates shape of the output of a function.
    """
    def decorator(shape_precalculator):
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





# @register_out_shape(np.sum)
# def numpy_sum_out_shape(X,