class InternalError(Exception):
    pass


class UnreachableError(InternalError):
    def __init__(self, msg: str = "Should never get here."):
        super().__init__(msg)


class DimensionMismatch(Exception):
    pass