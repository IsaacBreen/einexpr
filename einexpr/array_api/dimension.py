from abc import ABC, abstractmethod
from typing import Iterable, Optional, Sequence
from dataclasses import dataclass, field
import einexpr.utils as utils


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


class Dimension(DimensionObject):
    """
    Represents a single dimension of a tensor.
    """
    size: int = field(default=None)


@dataclass(frozen=True, eq=True, order=True)
class NamedDimension(Dimension):
    """
    A named dimension that binds eagerly to other dimensions of the same name.
    """
    
    name: str
    size: int = field(default=None)

    def with_size(self, size: int) -> "NamedDimension":
        return NamedDimension(self.name, size)

    def __str__(self):
        return self.name
    



@dataclass(frozen=True, eq=False)
class PositionalDimension(Dimension):
    """
    A dimension that is bound to a position in the tensor shape.
    """

    size: int = field(default=None)

    def with_size(self, size: int) -> "PositionalDimension":
        return PositionalDimension(size)

    def __str__(self):
        return "_"

    def __eq__(self, other):
        raise TypeError("PositionalDimension cannot be directly compared.")


class DimensionTuple(DimensionObject):
    """
    A tuple of dimensions.
    """

    def __init__(self, dimensions: Iterable[Dimension] = ()):
        self.dimensions = tuple(dimensions)

    def __str__(self):
        return " ".join(str(dimension) for dimension in self.dimensions)

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


class DimensionReplacement(DimensionObject):
    """
    A dimension that is to be replaced by another dimension.
    """
    size: int = field(default=None)

    def __init__(self, original: Dimension, replacement: Dimension):
        self.original = original
        self.replacement = replacement

    def __str__(self):
        original_str = utils.minimalistic_str(self.original, outer_brackets=False)
        replacement_str = utils.minimalistic_str(self.replacement, outer_brackets=False)
        return f"({original_str} -> {replacement_str})"
    
    def __repr__(self):
        return f"DimensionReplacement({self.original!r}, {self.replacement!r})"

    def __hash__(self):
        raise NotImplementedError("DimensionReplacement is not hashable.")
        # return hash((self.original, self.replacement))

    def __eq__(self, other):
        raise NotImplementedError("DimensionReplacement is not comparable.")
        # return (
        #     isinstance(other, DimensionReplacement)
        #     and self.original == other.original
        #     and self.replacement == other.replacement
        # )


Dimensions = Sequence[Dimension]

__all__ = [
    "DimensionObject",
    "Dimension",
    "Dimensions",
    "NamedDimension",
    "DimensionTuple",
    "DimensionReplacement",
]