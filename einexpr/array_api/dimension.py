from abc import ABC, abstractmethod
from typing import Sequence


class Dimension(ABC):
    """
    Represents a single dimension of a tensor.
    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass


class NamedBindingDimension(Dimension):
    """
    A named dimension that binds eagerly to other dimensions of the same name.
    """
    
    def __init__(self, name: str, size: int = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.size = size

    def __str__(self):
        return self.name

    def __repr__(self):
        return f'NamedBindingDimension({self.name})'

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, NamedBindingDimension) and self.name == other.name and self.size == other.size


class AnonymousDimension(Dimension):
    """
    A dimension without a name that does not bind to other dimensions (unless forced to).
    """
    
    def __init__(self, size: int = None, *args, **kwargs):
        self.size = size
        super().__init__(*args, **kwargs)

    def __str__(self):
        return '?'

    def __repr__(self):
        return 'AnonymousDimension()'

    def __hash__(self):
        return hash('AnonymousDimension')

    def __eq__(self, other):
        return isinstance(other, AnonymousDimension) and self.size == other.size
    
    
class PositionalDimension(Dimension):
    """
    A dimension that is bound to a position in the tensor shape.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __str__(self):
        return '_'

    def __repr__(self):
        return 'PositionalDimension()'

    def __hash__(self):
        return hash('PositionalDimension')

    def __eq__(self, other):
        raise TypeError('PositionalDimension cannot be compared.')


def DimensionReplacement


Dimensions = Sequence[Dimension]

__all__ = ['Dimension', 'Dimensions', 'NamedBindingDimension', 'AnonymousDimension', 'PositionalDimension']