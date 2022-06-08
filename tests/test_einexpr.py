from einexpr import __version__
from einexpr import einexpr, binary_arithmetic_operations, EinsteinExpression
import numpy as np
import jax
import jax.numpy as jnp
from collections import namedtuple
import string
import pprint
import pytest

pp = pprint.PrettyPrinter(indent=4)

np.random.seed(1234567890)

tolerance = 1e-3

def test_pow():
    w = einexpr(np.random.rand(4,2))
    x = einexpr(np.random.rand(2))
    expr = w['b,i'] * x['i']
    expr = expr['']
    print(expr)
    assert np.allclose(expr.eval(), np.einsum('bi,i->', w.eval(), x.eval()), tolerance)


RandomExpressionData = namedtuple("RandomExpressionData", ["expr", "expr_json", "var"])


@pytest.fixture
def make_random_expr(np_like, max_indices=8, max_index_size=5, max_vars=8, p_leaf=0.8):
    num_indices = np.random.randint(2, max_indices)
    num_indices_out = np.random.randint(0,num_indices)
    index_names = list(string.ascii_lowercase[:num_indices])
    index_names_out = list(string.ascii_lowercase[:num_indices_out])
    index_sizes_per_var = {i:s for i,s in zip(index_names, np.random.randint(1, max_index_size, size=num_indices))}

    n_vars = np.random.randint(2, max_vars)
    per_var_indices = [list(np.random.choice(index_names, size=np.random.randint(1, num_indices), replace=False)) for _ in range(n_vars)]
    vars = [np.random.rand(*(index_sizes_per_var[i] for i in var_indices)) for var_indices in per_var_indices]
    vars = [np_like.array(v) for v in vars]
    ops = binary_arithmetic_operations
    
    def _make_random_expr(vars, per_var_indices, index_names, ops, p_leaf):
        if np.random.rand() < p_leaf:
            var_num = np.random.randint(0, n_vars)
            var = vars[var_num]
            expr = einexpr(var)[",".join(per_var_indices[var_num])]
            expr_json = {"type": "leaf", "value": var, "indices": per_var_indices[var_num]}
            var_normshaped = np.einsum(f"{''.join(i for i in per_var_indices[var_num])}->{''.join(i for i in index_names if i in per_var_indices[var_num])}", var)
            expand_indices = [n for n, i in enumerate(index_names) if i not in per_var_indices[var_num]]
            var_normshaped = np.expand_dims(var_normshaped, expand_indices)
            return expr, expr_json, var_normshaped
        else:
            op = np.random.choice(list(ops))
            expr_lhs, expr_json_lhs, var_lhs = _make_random_expr(vars, per_var_indices, index_names, ops, p_leaf)
            expr_rhs, expr_json_rhs, var_rhs = _make_random_expr(vars, per_var_indices, index_names, ops, p_leaf)
            expr = getattr(expr_lhs, op)(expr_rhs)
            expr_json = {"type": "op", "op": op, "lhs": expr_json_lhs, "rhs": expr_json_rhs}
            var = getattr(var_lhs, op)(var_rhs)
            return expr, expr_json, var
        
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        expr, expr_json, var = _make_random_expr(vars, per_var_indices, index_names, ops, p_leaf)
        expr = expr[",".join(index_names_out)]
        var = np.sum(var, axis=tuple(n for n, i in enumerate(index_names) if i not in index_names_out))
    return RandomExpressionData(expr, expr_json, var)


# @pytest.mark.skip
@pytest.mark.parametrize("i", range(100))
@pytest.mark.parametrize("np_like", [np, jnp])
def test_random_expr(i, np_like, make_random_expr):
    expr, expr_json, var = make_random_expr
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        if not np.allclose(expr.eval(), var, tolerance) and np.all([~np.isnan(expr.eval()), ~np.isnan(var)]):
            print(expr)
            pp.pprint(expr_json)
            print(expr.get_shape())
            print(var.shape)
            raise ValueError(f"Values do not match: {expr.eval()} != {var}")


# @pytest.mark.skip
@pytest.mark.parametrize("i", range(100))
@pytest.mark.parametrize("np_like", [jnp])
def test_random_expr_jax_jit(i, make_random_expr):
    expr, expr_json, var = make_random_expr
    assert not any(x is None for x in make_random_expr), f"One of the expressions from {make_random_expr} is None"
    
    # @jax.jit
    def eval_expr(expr):
        return expr.eval()
    
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        assert eval_expr(expr) is not None
        if not np.allclose(eval_expr(expr), var, tolerance) and np.all([~np.isnan(eval_expr(expr)), ~np.isnan(var)]):
            print(expr)
            pp.pprint(expr_json)
            print(expr.get_shape())
            print(var.shape)
            raise ValueError(f"Values do not match: {eval_expr(expr)} != {var}")


def test_simple_expr_jax_jit():
    @jax.jit
    def add(x,y):
        x = einexpr(x)
        y = einexpr(y)
        z = x['i'] + y['i']
        return z[''].eval()
    x = np.random.rand(2)
    y = np.random.rand(2)
    add(x,y)

    @jax.jit
    def add_and_reduce(x,y):
        z = x['i'] + y['i']
        return z['']
    x = einexpr(np.random.rand(2))['i']
    y = einexpr(np.random.rand(2))['i']
    add_and_reduce(x,y)
