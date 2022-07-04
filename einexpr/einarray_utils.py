class EinarrayOperatorsMixin:
    # Binary operators
    for magic_name in "add radd sub rsub mul rmul div rdiv floordiv rfloordiv truediv rtruediv pow rpow eq".split():
        full_magic_name = f'__{magic_name}__'
        locals()[full_magic_name] = eval(f"lambda self, other:  self.__einarray_function__(getattr(type(self.a), '{full_magic_name}'), self, other)")
    # Unary operators
    for magic_name in "neg pos invert abs ceil floor round trunc floor divide conj".split():
        full_magic_name = f'__{magic_name}__'
        locals()[full_magic_name] = eval(f"lambda self:  self.__einarray_function__(getattr(type(self.a), '{full_magic_name}'), self)")
