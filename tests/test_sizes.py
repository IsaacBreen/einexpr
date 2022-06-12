import pytest
import numpy as np
from einexpr import calculate_output_shape
from einexpr import einexpr

@pytest.mark.skip
def test_output_shape_numpy_sum():
    MAX_DIMS = 10
    rng = np.random.default_rng()
    X_ndim = rng.integers(1, MAX_DIMS)
    X_shape = rng.random_integers(1, 10, size=(3))
    print(X_shape)
    
    # X = einexpr(np.ones((2, 3)))['i j']
    # assert calculate_output_shape(np.sum, X) == ()
    
