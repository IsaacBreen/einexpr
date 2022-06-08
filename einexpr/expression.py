from dataclasses import dataclass, field, InitVar
from typing import List, Tuple, Dict, Set, Any, Optional, Union
import string
import re
import math

import numpy as np
import jax.numpy as jnp
from jax.tree_util import register_pytree_node, register_pytree_node_class

from .utils import deprecated
from .index import Index


multiplicitave_operations = set(["__mul__", "__rmul__", "__truediv__", "__rtruediv__"])
additive_operations = set(["__add__", "__radd__", "__sub__", "__rsub__"])
power_operations = set(["__pow__", "__rpow__"])
binary_arithmetic_operations = multiplicitave_operations | additive_operations | power_operations

array_types = (np.ndarray, np.generic, jnp.ndarray)

EAGER_MODE = True

def get_np_like_interface(obj):
    """
    Returns the np-like interface of the given object.
    """
    if isinstance(obj, list):
        for obj in obj:
            try:
                return get_np_like_interface(obj)
            except TypeError:
                pass
        raise TypeError(f"Cannot get np-like interface of {type(obj)}")
    module_name = obj.__class__.__module__
    if isinstance(obj, np.ndarray) or module_name == "numpy":
        return np
    elif isinstance(obj, jnp.ndarray) or module_name == "jaxlib.xla_extension":
        return jnp
    else:
        raise TypeError(f"Unrecognized type: {type(obj)}")

def intercept_binary_arithmetic(func, pass_existing=False):
    def decorator(cls):
        names = ["__add__", "__radd__", "__sub__", "__rsub__", "__mul__", "__rmul__", "__truediv__", "__rtruediv__", "__pow__", "__rpow__"]
        for name in names:
            if pass_existing:
                setattr(cls, name, func(name, getattr(cls, name) if hasattr(cls, name) else None))
            else:
                setattr(cls, name, func)
        return cls
    return decorator


def parse_indices(indices):
    """
    Parses an index string such as into a list of Index objects. E.g. 'x,y[0],y[1]' -> [Index('x'), Index('y', 0), Index('y', 1)]
    """
    if isinstance(indices, str):
        indices = re.split(",| ", indices)
    indices = [index2 for index1 in indices for index2 in re.split(",| ", index1)]
    indices2 = []
    for i in indices:
        if not i:
            continue
        if '[' in i:
            i = i.split('[')
            indices2.append(Index(i[0], i[1][:-1]))
        else:
            indices2.append(Index(i))
    return indices2


def compute_transpose(shape, target_shape):
    transposition = []
    for i in target_shape:
        if i in shape:
            transposition.append(shape.index(i))
        else:
            transposition.append(np.newaxis)
    return tuple(transposition)


def transpose_and_expand(a, axes):
    """
    Transpose the given array to the given axes, expanding wherever a `None` axis is encountered.
    """
    transpose_indices = [i for i in axes if i is not None]
    expand_np_index = tuple(None if i is None else slice(None) for i in axes)
    return a.transpose(transpose_indices).__getitem__(expand_np_index)


def has_definite_shape(value):
    """
    Returns True if the given object has a defined size, False otherwise. If the given value has indices, returns True, and otherwise returns False.
    Also returns True for zero-dimensional values such as int and float.
    """
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, (EinsteinExpression, Shaped)):
        return True
    if hasattr(value, "get_shape"):
        try:
            value.get_shape()
            return True
        except AssertionError:
            return False
    return False


def get_indices(x):
    """
    Returns the indices of the given object.
    """
    if isinstance(x, (int, float)):
        return set()
    elif isinstance(x, EinsteinObject):
        return x.get_inner_indices()
    else:
        raise TypeError(f"Unrecognized type: {type(x)}")


def can_trickle(obj):
    """
    Returns whether the given object "trickle down" to its args.
    """
    if isinstance(obj, EinsteinTensor):
        return False
    if isinstance(obj, EinsteinExpression):
        return True
    return False


def einexpr(x):
    """
    Returns the given object as an EinsteinExpression if it is not already one.
    """
    return unique_einexpr(x)


def einfunc(func):
    """
    A decorator that converts function arguments into EinsteinExpressions.
    """
    def wrapper(*args, **kwargs):
        args = [einexpr(arg) if isinstance(arg, array_types) else arg for arg in args]
        kwargs = {k: einexpr(v) if isinstance(v, array_types) else v for k, v in kwargs.items()}
        return func(*args, **kwargs)
    return wrapper


@intercept_binary_arithmetic(lambda name, _: lambda self, other: EinsteinExpression(name, self, other), pass_existing=True)
class EinsteinObject:
    def copy(self):
        raise NotImplementedError(f"{self.__class__.__name__}")
    
    def eq(self, other):
        raise NotImplementedError(f"{self.__class__.__name__}")
        
    def is_initialized(self):
        raise NotImplementedError(f"{self.__class__.__name__}")
    
    def get_inner_indices(self):
        raise NotImplementedError(f"{self.__class__.__name__}")
    
    @deprecated
    def get_adjacent_indices(self):
        raise NotImplementedError(f"{self.__class__.__name__}")
    
    def get_outer_indices(self):
        raise NotImplementedError(f"{self.__class__.__name__}")
    
    @deprecated
    def get_lower_indices(self):
        return self.get_inner_indices()
    
    def get_mul_indices(self):
        raise NotImplementedError(f"{self.__class__.__name__}")
    
    def get_dim_sizes(self):
        raise NotImplementedError(f"{self.__class__.__name__}")
    
    def eq(self):
        raise NotImplementedError(f"{self.__class__.__name__}")
    
    def __getitem__(self, shape):
        return Shaped(self, parse_indices(shape)).__getitem__(shape)
        
    def __repr__(self):
        raise NotImplementedError(f"{self.__class__.__name__}")
    
    def debug_dict(self):
        raise NotImplementedError(f"{self.__class__.__name__}")
    
    @deprecated
    @staticmethod
    def is_tricklable(self=None):
        raise NotImplementedError(f"{self.__class__.__name__}")


@dataclass()
@register_pytree_node_class
class EinsteinExpression(EinsteinObject):
    operation: InitVar[str]
    args: List[Any] = field(init=False)
    inner_indices: Set[str] = field(init=False, default_factory=set)
    outer_indices: Set[str] = field(init=False, default_factory=set)
    mul_indices: Set[str] = field(init=False, default_factory=set)
    shape: Tuple[str, ...] = None
    dim_sizes: Dict[str, int] = None

    def __init__(self, operation, *args):
        self.operation = operation
        for arg in args:
            assert has_definite_shape(arg), f"Argument {arg} must be shaped (indexed) before it can be used in an EinsteinExpression, e.g. x[a,b,c]*2 instead of x*2"
        self.args = [unique_einexpr(arg) for arg in args]
        self.inner_indices = {i for arg in self.args for i in arg.get_inner_indices()}
        self.outer_indices = set()
        self.mul_indices = {i for arg in self.args for i in arg.get_inner_indices()} if operation in multiplicitave_operations | power_operations else set()
        self.shape = None
        self.dim_sizes = {}
        for arg in self.args:
            assert all(self.get_dim_sizes()[i] == arg.get_dim_sizes()[i] for i in self.dim_sizes.keys() & arg.get_dim_sizes().keys()), f"Dimension sizes must be the same for all arguments. Got {self.get_dim_sizes()} and {arg.get_dim_sizes()}"
            self.dim_sizes |= arg.get_dim_sizes()
            
    def __array__(self, dim_sizes=None):
        if dim_sizes is None:
            dim_sizes = self.get_dim_sizes()
        return getattr(self.args[0].__array__(dim_sizes=dim_sizes), self.operation)(*(arg.__array__(dim_sizes=dim_sizes) for arg in self.args[1:]))
    
    @staticmethod
    def broadcast_args(args):
        # Broadcast the shapes of the arguments
        shapes = [arg.get_shape() for arg in args]
        resolved_shape = []
        for shape in shapes:
            for i in shape:
                if i not in resolved_shape:
                    resolved_shape.append(i)
        resolved_shape = tuple(resolved_shape)
        transposed_args = [Transposition(arg, resolved_shape) if resolved_shape != arg.get_shape() else arg for arg in args]
        return transposed_args
    
    def coerce_into_shape(self, shape, multiply_indices=None, eager_transpose=False, additive_indices_that_collapse_in_parent=None):
        multiply_indices = multiply_indices or set()
        additive_indices_that_collapse_in_parent = additive_indices_that_collapse_in_parent or set()
        unexpanded_collapsing_indices = additive_indices_that_collapse_in_parent - self.get_inner_indices()
        if self.operation in multiplicitave_operations:
            args = self.args.copy()
            args = [arg.coerce_into_shape(shape, multiply_indices | self.get_mul_indices()) for arg in args]
            inner_indices = {i for arg in args for i in arg.get_inner_indices()}
            noncollapsable_indices = inner_indices & (set(shape) | multiply_indices)
            if broadcastable(args) and noncollapsable_indices == inner_indices:
                # Args are broadcastable and there's nothing to collape, so we don't need to use EinSum here.
                out_shape = calculate_broadcast_shape(args)
                new_expr = EinsteinExpression(self.operation, *args)
            else:
                # Otherwise, we do need to use EinSum.
                if self.operation in {"__truediv__", "__rtruediv__"}:
                    # We need to convert the expression from the form x/y to x*(1/y) to make it compatible with EinSum.
                    pos = 1 if self.operation == "__truediv__" else 0
                    args[pos] = 1/args[pos]
                arg_shapes = [arg.get_shape() for arg in args]
                out_shape = tuple(noncollapsable_indices)
                new_expr = EinSum(out_shape, arg_shapes, args)
            new_expr = Shaped(new_expr, out_shape, coerce=False)
            if unexpanded_collapsing_indices:
                new_expr = math.prod((new_expr, *(IndexSize(index) for index in unexpanded_collapsing_indices)))
            return new_expr
        else:
            initial_inner_indices = {i for arg in self.args for i in arg.get_inner_indices()}
            additive_collapsable_indices = initial_inner_indices - set(shape) - multiply_indices if self.operation in additive_operations else None
            args = self.args
            args = [arg.coerce_into_shape(shape, multiply_indices | self.get_mul_indices(), additive_indices_that_collapse_in_parent=additive_collapsable_indices) for arg in args]
            args = self.broadcast_args(args)
            new_expr = EinsteinExpression(self.operation, *args)
            remaining_inner_indices = new_expr.get_inner_indices()
            remaining_noncollapsable_indices = remaining_inner_indices & (set(shape) | multiply_indices)
            # Collapse on all indices except those that are involved in multiplication higher up in the expression and those in the target shape.
            if eager_transpose or remaining_noncollapsable_indices != remaining_inner_indices or unexpanded_collapsing_indices:
                # Collapse and reshape
                new_shape = tuple(remaining_noncollapsable_indices)
                args = [new_expr, *{IndexSize(index) for index in unexpanded_collapsing_indices}]
                new_expr = EinSum(new_shape, [arg.get_shape() for arg in args], args)
            else:
                new_shape = new_expr.get_shape()
            new_expr = Shaped(new_expr, new_shape, coerce=False)
            return new_expr
    
    def copy(self):
        return EinsteinExpression(self.operation, *[unique_einexpr(arg) for arg in self.args])
    
    def eq(self, other):
        return all((
            type(self) is type(other),
            self.operation == other.operation,
            len(self.args) == len(other.args),
            [type(arg) == type(other_arg) and arg.eq(other_arg) for arg, other_arg in zip(self.args, other.args)],
            self.inner_indices == other.inner_indices,
            self.outer_indices == other.outer_indices,
            self.mul_indices == other.mul_indices
        ))
    
    def get_inner_indices(self):
        return self.inner_indices
    
    @deprecated
    def get_adjacent_indices(self):
        return self.adjacent_indices
    
    def get_outer_indices(self):
        return self.outer_indices
    
    def get_mul_indices(self):
        return self.mul_indices
    
    def remove_mul_indices(self, indices, recursive=True, include_self=True):
        if include_self:
            self.mul_indices -= indices
        if recursive:
            for arg in self.args:
                if can_trickle(arg):
                    arg.remove_mul_indices(indices, recursive=recursive)
    
    def get_dim_sizes(self):
        return self.dim_sizes
    
    @deprecated
    @staticmethod
    def is_tricklable(self=None):
        return True

    def _calculate_shape(self):
        self.shape = calculate_broadcast_shape(self.args)        
    
    def get_shape(self):
        if self.shape is None:
            self._calculate_shape()
        return self.shape
        
    def __repr__(self):
        return f"{self.operation}({', '.join(map(repr, self.args))})"
    
    def debug_dict(self):
        return {
            "operation": self.operation,
            "args": [arg.debug_dict() if hasattr(arg, "debug_dict") else arg for arg in self.args],
            "inner_indices": list(self.get_inner_indices()),
            "outer_indices": list(self.get_outer_indices()),
            "mul_indices": list(self.get_mul_indices())
        }    

    def tree_flatten(self):
        # children = {"args": self.args}
        # aux_data = {"operation": self.operation}
        children = self.args
        aux_data = (self.operation,)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # return cls(aux_data["operation"], *children["args"])
        return cls(*aux_data, *children)


@intercept_binary_arithmetic(lambda name, _: lambda self, other: EinsteinExpression(name, self, other), pass_existing=True)
@register_pytree_node_class
class IndexSize:
    def __init__(self, index):
        assert isinstance(index, (str, Index))
        if isinstance(index, Index):
            self.index = index.get_base()
        else:
            self.index = index
        
    def get_shape(self):
        return ()
    
    def get_inner_indices(self):
        return set()
    
    def get_dim_sizes(self):
        return {}
    
    def coerce_into_shape(self, *args, **kwargs):
        return self
    
    def copy(self):
        return self
    
    def __array__(self, dim_sizes):
        if self.index in dim_sizes:
            return dim_sizes[self.index]
        else:
            raise KeyError(f"Index {self.index} not found in dim_sizes {dim_sizes}")

    def __repr__(self):
        return f'IndexSize({self.index})'
    
    def __eq__(self, other):
        return type(self) == type(other) and self.index == other.index

    def __hash__(self):
        return hash(self.index)

    def tree_flatten(self):
        children = None
        aux_data = (self.index,)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*aux_data)


@intercept_binary_arithmetic(lambda name, _: lambda self, other: EinsteinExpression(name, self, other), pass_existing=True)
@register_pytree_node_class
class Transposition:
    """
    A class that represents a transposition and/or expansion of an EinsteinObject.
    """
    def __init__(self, value, to):
        self.value = value
        self.shape = to
        self.permutation = tuple(value.get_shape().index(index) if index in value.get_shape() else np.newaxis for index in to)
        
    def __array__(self, dim_sizes=None):
        return transpose_and_expand(self.value.__array__(dim_sizes=dim_sizes), self.permutation)
    
    def __getitem__(self, shape):
        return Shaped(self, parse_indices(shape)).__getitem__(shape)
    
    def copy(self):
        return Transposition(self.value.copy(), self.shape)
        
    def __repr__(self):
        return f'{self.__class__.__name__}({self.value!r}, {self.shape!r})'
    
    def __getattr__(self, name):
        return getattr(self.value, name)
    
    def debug_dict(self):
        return {'value': self.value.debug_dict(), 'shape': self.shape, 'permutation': self.permutation}
    
    def get_shape(self):
        return self.shape
    
    def get_dim_sizes(self):
        return self.value.get_dim_sizes()

    def tree_flatten(self):
        children = (self.value,)
        aux_data = (self.shape,)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children[0], aux_data[0])


def unique_einexpr(x):
    if isinstance(x, (int, float)):
        return Shaped(EinsteinTensor(x), ())
    elif isinstance(x, array_types):
        return EinsteinTensor(x)
    elif isinstance(x, (EinsteinObject, Shaped, Transposition)):
        return x.copy()
    elif isinstance(x, IndexSize):
        return x
    else:
        raise TypeError(f"Unrecognized type: {type(x)}")

@register_pytree_node_class
class EinSum(EinsteinObject):
    """
    A class that represents a sum of EinsteinObjects.
    """
    def __init__(self, target_shape, arg_shapes, args):
        assert isinstance(target_shape, (list, tuple))
        assert all(isinstance(i, Index) for i in target_shape)
        assert isinstance(arg_shapes, (list, tuple))
        assert all(isinstance(shape, (list, tuple)) for shape in arg_shapes)
        assert isinstance(args, (list, tuple))
        assert set(target_shape).issubset(set.union(*(set(shape) for shape in arg_shapes))), f"Argument shapes must be a subset of the target shape. Got extra indices: {set(target_shape) - set.union(*(set(shape) for shape in arg_shapes))}."
        self.target_shape = target_shape
        self.arg_shapes = arg_shapes
        self.args = args

    def get_einstr(self):
        arg_indices = {i for shape in self.arg_shapes for i in shape}
        available_lowercase_letters = set(string.ascii_lowercase)
        index_map = {}
        for index in arg_indices:
            if (i := index.get_base()[0]) in available_lowercase_letters:
                index_map[index] = i
                available_lowercase_letters.remove(i)
            else:
                index_map[index] = available_lowercase_letters.pop()
        arg_shape_einstrs = ["".join(index_map[i] for i in shape) for shape in self.arg_shapes]
        arg_shape_einstr = ",".join(arg_shape_einstrs)
        out_shape_einstr = "".join(index_map[i] for i in self.target_shape)
        return f"{arg_shape_einstr}->{out_shape_einstr}"
    
    def __array__(self, dim_sizes=None):
        arg_evals = [arg.__array__(dim_sizes=dim_sizes) for arg in self.args]
        return get_np_like_interface(arg_evals).einsum(self.get_einstr(), *arg_evals)
    
    def copy(self):
        return EinSum(self.target_shape, self.arg_shapes, [arg.copy() for arg in self.args])
    
    def __repr__(self):
        return f'{self.__class__.__name__}({self.get_einstr()!r}, {self.args!r})'
    
    def debug_dict(self):
        return {'target_shape': self.target_shape, 'arg_shapes': self.arg_shapes, 'args': [arg.debug_dict() for arg in self.args], 'einstr': self.get_einstr()}
    
    def eq(self, other):
        if not isinstance(other, EinSum):
            return False
        if self.sum_str != other.sum_str:
            return False
        if len(self.args) != len(other.args):
            return False 
        for arg, other_arg in zip(self.args, other.args):
            if not arg.eq(other_arg):
                return False
        return True
    
    def get_dim_sizes(self):
        return {i: s for arg in self.args for i, s in arg.get_dim_sizes().items()}

    def tree_flatten(self):
        children = (self.args,)
        aux_data = (self.target_shape, self.arg_shapes)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(aux_data[0], aux_data[1], children[0])


def calculate_broadcast_shape(args):
    """
    Calculate the broadcast shape of the given arguments.
    """
    shape = []
    for arg in args:
        for n, i in enumerate(reversed(arg.get_shape())):
            if n >= len(shape):
                shape.append(i)
            if shape[n] != i and i is not np.newaxis:
                # If the index of the arg does not match the corresponding index of the inferred shape, the args are not broadcastable.
                raise ValueError(f"Cannot broadcast from {shape} to {arg.get_shape()}")
            # If shape[n] is None, overwrite it with i
            shape[n] = shape[n] or i
    return tuple(reversed(shape))
    
    
def broadcastable(args):
    """
    Check if the given arguments are broadcastable.
    """
    try:
        calculate_broadcast_shape(args)
        return True
    except ValueError:
        return False


@intercept_binary_arithmetic(lambda name, _: lambda self, other: EinsteinExpression(name, self, other), pass_existing=True)
@register_pytree_node_class
class Shaped:
    """
    A class that represents a shaped object.
    """
    def __init__(self, value, shape, coerce=True):
        if isinstance(value, EinsteinTensor):
            assert value.ndims() == len(shape), f"Tensor has {value.ndims()} dimensions, but shape has {len(shape)}"
        self.shape = tuple(shape)
        if has_definite_shape(value) and coerce:
            self.value = value.coerce_into_shape(self.shape)
            if self.shape != self.value.get_shape():
                self.value = Transposition(self.value, self.shape)
            assert set(self.value.get_shape()) == set(self.shape), f"Shape mismatch: {self.value.get_shape()} != {self.shape}"
            self.shape = self.value.get_shape()
            assert self.shape is not None, "Shape cannot be None"
        else:
            self.value = value
        if isinstance(self.value, Shaped) and self.value.get_shape() == self.shape:
            self.value = self.value.value
            
    def __getitem__(self, shape):
        value = Shaped(self, parse_indices(shape))
        return Shaped(einexpr(value.__array__()), value.get_shape(), coerce=False)
        
    def copy(self):
        return Shaped(self.value.copy(), self.shape)
        
    def __repr__(self):
        return f'{self.__class__.__name__}({self.value!r}, {self.shape!r})'
    
    def __str__(self):
        index_str = ", ".join(f"'{i}'" for i in self.shape)
        return f"{self.value}[{index_str}]"
    
    def __getattr__(self, name):
        return getattr(self.value, name)
    
    def get_inner_indices(self):
        return set(self.shape)
    
    @deprecated
    def get_adjacent_indices(self):
        return set()
    
    def debug_dict(self):
        return {'value': self.value.debug_dict(), 'shape': self.shape}
    
    def get_mul_indices(self):
        return set(self.shape)
    
    def get_dim_sizes(self):
        if isinstance(self.value, EinsteinTensor):
            return {i.get_base(): s for i, s in zip(self.shape, self.value.get_numeric_shape())}
        else:
            return self.value.get_dim_sizes()
    
    def get_shape(self):
        return self.shape
    
    def coerce_into_shape(self, shape, multiply_indices=None, additive_indices_that_collapse_in_parent=None, **kwargs):
        """
        Collapse unused dimensions. Do not attempt to expand dimensions - leave that to explicit broadcast methods.
        """
        multiply_indices = multiply_indices or set()
        additive_indices_that_collapse_in_parent = additive_indices_that_collapse_in_parent or set()
        unexpanded_collapsing_indices = additive_indices_that_collapse_in_parent - self.get_inner_indices() - multiply_indices
        # The collapsable indices are those that are not in the target shape and are not later involved in multiplication.
        collapsable_indices = set(self.get_shape()) - set(shape) - multiply_indices
        out_shape = tuple(i for i in self.get_shape() if i not in collapsable_indices)
        if out_shape != self.get_shape() or unexpanded_collapsing_indices:
            sizes_of_collapsing_indices = {IndexSize(index) for index in unexpanded_collapsing_indices}
            args = [self, *sizes_of_collapsing_indices]
            return Shaped(EinSum(out_shape, [arg.get_shape() for arg in args], [self, *sizes_of_collapsing_indices]), out_shape)
        else:
            return self
            
    def tree_flatten(self):
        children = (self.value,)
        aux_data = (self.shape,)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children[0], aux_data[0])



@register_pytree_node_class
class EinsteinTensor(EinsteinObject):
    """
    Wraps a tensor, which can be from a multitude of libraries (e.g. PyTorch, TensorFlow, JAX), in a form
    that allows for Einstein-like indexation.
    """
    def __init__(self, value):
        self.value = value
        
    def __array__(self, dim_sizes=None):
        return self.value
        
    def copy(self):
        return EinsteinTensor(self.value)
    
    def eq(self, other):
        return type(self) is type(other) and self.value == other.value
        
    def is_initialized(self):
        raise NotImplementedError
    
    def get_inner_indices(self):
        return set()
    
    @deprecated
    def get_adjacent_indices(self):
        return set()
    
    def get_outer_indices(self):
        return set()
    
    @deprecated
    @staticmethod
    def is_tricklable(self=None):
        return False
    
    def get_shape(self):
        assert isinstance(self.value, (int, float)), f"Shape of type {type(self.value)} is not supported"
        return ()
    
    def get_numeric_shape(self):
        if isinstance(self.value, (int, float)):
            return ()
        return self.value.shape
    
    def ndims(self):
        if isinstance(self.value, (int, float)):
            return 0
        return len(self.value.shape)
    
    def coerce_into_shape(self, shape, **kwargs):
        assert isinstance(self.value, (int, float)), f"Cannot coerce type {type(self.value)} into shape {shape}"
        return self
    
    def __getitem__(self, index):
        if isinstance(index, slice):
            raise NotImplementedError("Slicing is not yet supported.")
        # The tensor is being indexed for the first time. The first index defines the shape of the tensor.
        elif isinstance(index, list):
            raise NotImplementedError("List indexing is not yet supported.")
        elif isinstance(index, str):
            shape = parse_indices(index)
        elif isinstance(index, tuple):
            shape = index
        else:
            raise TypeError("Index must be an int, list, or tuple.")
        if isinstance(self.value, array_types):
            assert len(shape) == len(self.value.shape), f"Shape {shape} does not match tensor shape {self.value.shape}"
        return Shaped(self, shape)
        
    def __repr__(self):
        if isinstance(self.value, array_types):
            return f"ndarray({self.value.shape}, {self.value.dtype})"
        else:
            return repr(self.value)
    
    def debug_dict(self):
        return {
            "value": self.value
        }

    def tree_flatten(self):
        children = (self.value,)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
