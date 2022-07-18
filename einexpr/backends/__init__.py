from .backends import *
from .raw_ops import *


class PseudoRawArray:
    def __getattr__(self, name):
        return self