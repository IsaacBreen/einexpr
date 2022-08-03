import einexpr
from abc import ABC, abstractmethod, abstractproperty
from typing import Dict, Iterable, Optional, Sequence, Tuple, TypeVar, Union
from typing import overload
from dataclasses import dataclass, field
from uuid import uuid4

from pydantic import NegativeInt, PositiveInt
import einexpr.utils as utils


@utils.deprecated
class DimensionObject(ABC):

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass


@utils.deprecated
class _AtomicDimension(DimensionObject):
    """
    Represents a single dimension of a tensor.
    """
    size: int = field(default=None)
    
    @abstractproperty
    def id(self) -> str:
        pass


@utils.deprecated
@dataclass(frozen=True, eq=True)
class NamedDimension(_AtomicDimension):
    """
    A named dimension that binds eagerly to other dimensions of the same name.
    """
    
    name: str
    size: int = field(default=None)

    @property
    def id(self) -> str:
        return self.name

    def with_size(self, size: int) -> "NamedDimension":
        return NamedDimension(self.name, size)

    def __str__(self):
        return self.name
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NamedDimension):
            return False
        if self.name != other.name:
            return False
        if self.size is not None and self.size != other.size:
            return False
        return True
    
    def __hash__(self) -> int:
        return hash((self.name, self.size))


@utils.deprecated
def UniqueDimension():
    """
    A dimension object whose name is a UUID.
    """
    
    return NamedDimension('_' + str(uuid4()).replace('-', ''))


@dataclass(frozen=True, eq=False)
class AbsorbableDimension(_AtomicDimension):
    """
    A dimension that binds readily with other dimensions, regardless of their name.
    """
    
    size: PositiveInt = field(default=None)

    @property
    def id(self) -> str:
        return str(id(self))
    
    def __str__(self):
        return "*"
    
    def __eq__(self, other: object) -> bool:
        return self is other
    
    def __hash__(self) -> int:
        return hash(id(self))


@utils.deprecated
@dataclass(frozen=True, eq=False)
class PositionalDimension(_AtomicDimension):
    """
    A dimension that is bound to a position in the tensor.
    """

    position: NegativeInt
    size: Optional[PositiveInt] = field(default=None)
    
    def __post_init__(self):
        if self.position >= 0:
            raise ValueError("Position must be negative.")
        if self.size is not None and self.size < 0:
            raise ValueError("Size must be positive.")
    
    @property
    def id(self) -> str:
        return str(self.position)

    def with_size(self, size: int) -> "NamedDimension":
        return PositionalDimension(self.position, size)

    def __str__(self) -> str:
        return "_"
    
    @property
    def name(self) -> str:
        return str(self.position)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PositionalDimension):
            return False
        if self.position != other.position:
            return False
        if self.size is not None and self.size != other.size:
            return False
        return True
    
    def __hash__(self) -> int:
        return hash((self.position, self.size))


@utils.deprecated
class DimensionTuple(DimensionObject):
    """
    A tuple of dimensions.
    """

    def __init__(self, dimensions: Iterable[_AtomicDimension] = ()):
        self.dimensions = tuple(dimensions)

    def __str__(self):
        return '(' + " ".join(str(dimension) for dimension in self.dimensions) + ')'

    def __repr__(self):
        return f"DimensionTuple({self.dimensions})"

    def __hash__(self):
        return hash(self.__str__())
    
    def __eq__(self, other):
        if not isinstance(other, DimensionTuple):
            return False
        return self.dimensions == other.dimensions
    
    def __lt__(self, other):
        if isinstance(other, DimensionTuple):
            return self.dimensions < other.dimensions
        else:
            raise TypeError(f"Cannot compare {type(other)} to DimensionTuple")

    def __getitem__(self, index):
        return self.dimensions[index]

    def __len__(self):
        return len(self.dimensions)

    def __iter__(self):
        return iter(self.dimensions)

    def __contains__(self, item):
        return item in self.dimensions

    def __add__(self, other):
        if isinstance(other, DimensionTuple):
            return DimensionTuple(self.dimensions + other.dimensions)
        else:
            raise TypeError(f"Cannot add {type(other)} to DimensionTuple")

    def index(self, dimension):
        return self.dimensions.index(dimension)


@utils.deprecated
class DimensionReplacement(DimensionObject):
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


@utils.deprecated
class PositionalDimensionPlaceholder:
    pass


@overload
def dims(n: int) -> Tuple[NamedDimension, ...]:
    ...


@overload
def dims(size_by_name: Dict[str, int]) -> Tuple[NamedDimension, ...]:
    ...

D = TypeVar("D", bound=DimensionObject)
@overload
def dims(d: D) -> D:
    ...


@overload
def dims(s: str) -> Union[NamedDimension, DimensionTuple]:
    ...

def dims(x):
    if isinstance(x, int):
        return tuple(UniqueDimension() for _ in range(x))
    elif isinstance(x, dict) and all(isinstance(k, str) for k in x):
        return tuple(NamedDimension(k, v) for k, v in x.items())
    elif isinstance(x, (list, tuple, DimensionObject)):
        return einexpr.dimension_utils.parse_dims(x)
    elif isinstance(x, str):
        d = einexpr.dimension_utils.parse_dims(x)
        if len(d) == 1:
            return d[0]
        else:
            return d
    else:
        raise TypeError(f"Cannot parse {type(x)} as dimensions.")



Dimension = _AtomicDimension | Tuple['Dimension', ...]
AtomicDimension = _AtomicDimension | str

class DimensionSpecification:
    """
    A specification of the dimensions of a tensor.
    """
    
    dimensions: Tuple[Dimension, ...]
    sizes: Dict[AtomicDimension, int]

    def __init__(self, dimensions: Iterable[AtomicDimension], sizes: Dict[AtomicDimension, int]):
        self.dimensions = tuple(dimensions)
        self.sizes = sizes or {}
        def verify_dimensions(dimensions) -> bool:
            for dim in dimensions:
                if isinstance(dim, tuple):
                    if not verify_dimensions(dim):
                        return False
                elif not isinstance(dim, AtomicDimension):
                    return False
            return True
        if not verify_dimensions(self.dimensions):
            raise ValueError("Dimensions must be a nexted tuple of AtomicDimensions.")
        if any(not isinstance(k, AtomicDimension) for k in self.sizes):
            raise ValueError("Sizes must be a dictionary of AtomicDimensions or strings.")
    
    @property
    def shape(self) -> Tuple[int, ...]:
        def deep_product(value: Tuple | int) -> int:
            if isinstance(value, AtomicDimension):
                if value in self.sizes:
                    return self.sizes[value]
                else:
                    raise ValueError(f"Size of dimension {value} not specified.")
            else:
                product = 1
                for v in value:
                    product *= deep_product(v)
                return product
        return tuple(deep_product(v) for v in self.dimensions)
    
    def __iter__(self):
        return iter(self.dimensions)
    
    def __len__(self):
        return len(self.dimensions)
    
    def __getitem__(self, index):
        return self.dimensions[index]
    
    def index(self, dimension):
        return self.dimensions.index(dimension)
    
    def __hash__(self):
        return hash((self.dimensions, tuple(self.sizes.items())))
    
    def __eq__(self, other):
        return isinstance(other, DimensionSpecification) and self.dimensions == other.dimensions and self.sizes == other.sizes

    def __str__(self):
        return str(self.dimensions)
    
    def __repr__(self):
        return f"DimensionSpecification({self.dimensions!r}, {self.sizes!r})"
    
    def copy(self):
        return DimensionSpecification(self.dimensions, self.sizes.copy())


__all__ = [
    "DimensionObject",
    "AtomicDimension",
    "_AtomicDimension",
    "DimensionSpecification",
    "NamedDimension",
    "PositionalDimension",
    "UniqueDimension",
    "AbsorbableDimension",
    "DimensionTuple",
    "DimensionReplacement",
    "PositionalDimensionPlaceholder",
    "dims",
    "Dimension",
    "DimensionSpecification",
]