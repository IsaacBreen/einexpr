import numpy as np
from einexpr import calculate_output_shape
from einexpr import einexpr


def test_output_shape_numpy_sum():
    X = einexpr(np.ones((2, 3)))['i j']
    assert calculate_output_shape(np.sum, X) == ()
    
