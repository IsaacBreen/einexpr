import numpy as np
from einexpr import *
from einexpr import calculate_output_dims_from_signature


def test_calculate_output_dims_from_signature_and_einarrays():
    # Test 1: element-wise
    signature = parse_ufunc_signature("(),()->()")
    in_dims = [('i', 'j'), ('j', 'k')]
    out_dims = calculate_output_dims_from_signature(signature, *in_dims)
    assert set(out_dims) == set('ijk')
    
    # Test 2: matmul
    signature = parse_ufunc_signature(np.matmul.signature)
    in_dims = [('i', 'j'), ('j', 'k')]
    out_dims = calculate_output_dims_from_signature(signature, *in_dims)
    assert set(out_dims) == {'i', 'k'}
    
    # Test 3: matmul with broadcast
    signature = parse_ufunc_signature(np.matmul.signature)
    in_dims = [('i', 'k', 'l'), ('j', 'k', 'l', 'm')]
    out_dims = calculate_output_dims_from_signature(signature, *in_dims)
    assert set(out_dims) == {'i', 'j', 'k', 'm'}