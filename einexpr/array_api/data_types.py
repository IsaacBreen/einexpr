from ._types import dtype as _dtype


py_bool = bool

class dtype:
    def __eq__(self: 'dtype', other: 'dtype', /) -> py_bool:
        """
        Computes the truth value of ``self == other`` in order to test for data type object equality.

        Parameters
        ----------
        self: dtype
            data type instance. May be any supported data type.
        other: dtype
            other data type instance. May be any supported data type.

        Returns
        -------
        out: bool
            a boolean indicating whether the data type objects are equal.
        """
        return self is other


class bool(dtype):
    """
    Boolean (``True`` or ``False``)
    """


class int8(dtype):
    """
    An 8-bit signed integer whose values exist on the interval ``[-128, +127]``.
    """


class int16(dtype):
    """
    A 16-bit signed integer whose values exist on the interval ``[−32,767, +32,767]``.
    """


class int32(dtype):
    """
    A 32-bit signed integer whose values exist on the interval ``[−2,147,483,647, +2,147,483,647]``.
    """


class int64(dtype):
    """
    A 64-bit signed integer whose values exist on the interval ``[−9,223,372,036,854,775,807, +9,223,372,036,854,775,807]``.
    """


class uint8(dtype):
    """
    An 8-bit unsigned integer whose values exist on the interval ``[0, +255]``.
    """


class uint16(dtype):
    """
    A 16-bit unsigned integer whose values exist on the interval ``[0, +65,535]``.
    """


class uint32(dtype):
    """
    A 32-bit unsigned integer whose values exist on the interval ``[0, +4,294,967,295]``.
    """


class uint64(dtype):
    """
    A 64-bit unsigned integer whose values exist on the interval ``[0, +18,446,744,073,709,551,615]``.
    """


class float32(dtype):
    """
    IEEE 754 single-precision (32-bit) binary floating-point number (see IEEE 754-2019).
    """


class float64(dtype):
    """
    IEEE 754 double-precision (64-bit) binary floating-point number (see IEEE 754-2019).
    """


__all__ = ['dtype', 'bool', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'float32', 'float64']