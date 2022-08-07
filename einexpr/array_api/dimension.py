import einexpr
from abc import ABC, abstractmethod, abstractproperty
from typing import Dict, Iterable, Iterator, Optional, Sequence, Tuple, TypeVar, Union
from typing import overload
from dataclasses import dataclass, field
from uuid import uuid4

from pydantic import NegativeInt, PositiveInt
import einexpr.utils as utils


@utils.deprecated
class _AtomicDimension:
    """
    Represents a single dimension of a tensor.
    """


class AbsorbingDimension:
    """
    A dimension that binds readily with other dimensions, regardless of their name.
    """
        
    def __str__(self):
        return "*"
    
    def __repr__(self) -> str:
        # Represent using memory address
        return f"<{type(self).__name__} at {id(self)}>"
    
    def __eq__(self, other: object) -> bool:
        return self is other
    
    def __hash__(self) -> int:
        return hash(id(self))


class DimensionReplacement:
    """
    A dimension that is to be replaced by another dimension.
    """
    size: int = field(default=None)

    def __init__(self, original: _AtomicDimension, replacement: _AtomicDimension):
        self.original = original
        self.replacement = replacement

    def __str__(self):
        # original_str = utils.minimalistic_str(self.original, outer_brackets=False)
        # replacement_str = utils.minimalistic_str(self.replacement, outer_brackets=False)
        original_str = str(self.original)
        replacement_str = str(self.replacement)
        return f"({original_str} -> {replacement_str})"
    
    def __repr__(self):
        return f"DimensionReplacement({self.original!r}, {self.replacement!r})"

    def __hash__(self):
        return hash((self.original, self.replacement))
        # raise NotImplementedError("DimensionReplacement is not hashable.")

    def __eq__(self, other):
        return isinstance(other, DimensionReplacement)
        # raise NotImplementedError("DimensionReplacement is not comparable.")
        # return (
        #     isinstance(other, DimensionReplacement)
        #     and self.original == other.original
        #     and self.replacement == other.replacement
        # )


NestedDimension = _AtomicDimension | Tuple['NestedDimension', ...]
BaseDimension = _AtomicDimension | AbsorbingDimension | Tuple['NestedDimension', ...]  # TODO: Most usages of NestedDimension throughout this repo should actually be BaseDimension. Fix this.
ConcreteDimension = _AtomicDimension | Tuple['ConcreteDimension', ...]
AtomicDimension = _AtomicDimension | str

UnparsedDimensions = ConcreteDimension


@overload
def dims(n: int) -> Tuple[ConcreteDimension, ...]:
    ...


@overload
def dims(n: UnparsedDimensions) -> Tuple[ConcreteDimension, ...]:
    ...


def dims(x) -> Tuple[ConcreteDimension, ...]:
    if isinstance(x, int):
        return tuple(str('_' + str(uuid4()).replace('-', '')) for _ in range(x))
    elif isinstance(x, (str, tuple, list)):
        return einexpr.dimension_utils.parse_dims(x)
    else:
        raise TypeError(f"Cannot parse {type(x)} as dimensions.")


@dataclass(frozen=True, init=True, eq=True)
class DimensionSpecification:
    """
    A specification of the dimensions of a tensor.
    """
    
    dimensions: Tuple[NestedDimension, ...]
    sizes: Dict[AtomicDimension, int] = field(default_factory=dict)

    def __post_init__(self):
        super().__setattr__('dimensions', tuple(self.dimensions))
        def verify_dimensions(dimensions) -> bool:
            for dim in dimensions:
                if isinstance(dim, tuple):
                    if not verify_dimensions(dim):
                        return False
                elif not isinstance(dim, (AtomicDimension, AbsorbingDimension)):
                    return False
            return True
        if not verify_dimensions(self.dimensions):
            raise ValueError("Dimensions must be a nested tuple of AtomicDimensions.")
        # if any(not isinstance(k, AtomicDimension | AbsorbingDimension) or not isinstance(v, int) for k,v in self.sizes.items()):
        #     raise ValueError("Sizes must be a dictionary of AtomicDimensions or strings.")
        if not all(utils.tree_contains(self.dimensions, dim) for dim in self.sizes.keys()):
            raise ValueError("Each dimension in sizes must also be present in the dimension structure.")
        # Ensure the size is fully specified.
        if unsized := self.get_unsized_dims():
            raise ValueError(f"Some dimension sizes, {unsized}, are not calculable from the available dimension sizes, {self.sizes}.")
    
    @property
    def shape(self) -> Tuple[int, ...]:
        def deep_product(value: Tuple | int) -> int:
            if isinstance(value, AtomicDimension | AbsorbingDimension):
                if value in self.sizes:
                    return self.sizes[value]
                else:
                    raise ValueError(f"Size of dimension {value} not specified.")
            else:
                product = 1
                for v in value:
                    product *= deep_product(v)
                return product
        return tuple(self.compute_dimension_size(dim) for dim in self.dimensions)
    
    def with_dimensions(self, dimensions: Iterable[AtomicDimension]) -> 'DimensionSpecification':
        """
        Return a new DimensionSpecification with this object's dimensions replaced by the given dimensions in-order. The lengths of the current and new dimensions must match. The sizes of the new dimensions are taken from the current object.
        """
        if isinstance(dimensions, Iterator):
            dimensions = tuple(dimensions)
        if len(dimensions) != len(self.dimensions):
            raise ValueError("Dimensions must have the same length.")
        # Replace the keys in the sizes dict with the new dimensions.
        sizes = {}
        for old_dim, new_dim in zip(self.dimensions, dimensions):
            if self.can_compute_dimension_size(new_dim):
                for dim in self.get_dimension_size_components(new_dim):
                    sizes[dim] = self.sizes[dim]
            else:
                sizes[new_dim] = self.compute_dimension_size(old_dim)
        return DimensionSpecification(dimensions, sizes)

    def get_dimension_size_components(self, dimension: BaseDimension) -> Tuple[BaseDimension]:
        """
        Return the items in `self.sizes` necessary to compute the size of the given dimension.
        """
        if isinstance(dimension, tuple):
            size_components = tuple(component for dim in dimension for component in self.get_dimension_size_components(dim))
            return self.sizes.get(dimension) if None in size_components else size_components
        elif dimension in self.sizes:
            return (dimension,)
        else:
            return None

    def compute_dimension_size(self, dimension: BaseDimension) -> int:
        """
        Compute the size of a dimension.
        """
        product = 1
        for dim in self.get_dimension_size_components(dimension):
            product *= self.sizes[dim]
        return product
    
    def can_compute_dimension_size(self, dimension: BaseDimension) -> bool:
        """
        Return whether the size of a dimension can be computed.
        """
        if dimension in self.sizes:
            return True
        elif isinstance(dimension, tuple):
            return all(self.can_compute_dimension_size(d) for d in dimension)
        else:
            return False

    def get_unsized_dims(self) -> Tuple[AtomicDimension | AbsorbingDimension, ...]:
        def _get_unsized_dims(dims) -> Iterator[AtomicDimension | AbsorbingDimension]:
            if isinstance(dims, AtomicDimension | AbsorbingDimension):
                if dims in self.sizes:
                    return
                else:
                    yield dims
            else:
                for v in dims:
                    if v not in self.sizes:
                        yield from _get_unsized_dims(v)
        return tuple(_get_unsized_dims(self.dimensions))
    
    def __iter__(self):
        return iter(self.dimensions)
    
    def __len__(self):
        return len(self.dimensions)
    
    def __getitem__(self, index):
        return self.dimensions[index]
    
    def index(self, dimension):
        return self.dimensions.index(dimension)
    
    def __repr__(self):
        return f"DimensionSpecification({self.dimensions!r}, {self.sizes!r})"
    
    def copy(self):
        return DimensionSpecification(self.dimensions, self.sizes.copy())


__all__ = [
    "AtomicDimension",
    "_AtomicDimension",
    "DimensionSpecification",
    "AbsorbingDimension",
    "DimensionReplacement",
    "dims",
    "NestedDimension",
    "BaseDimension",
    "DimensionSpecification",
    "ConcreteDimension",
]