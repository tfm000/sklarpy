# Contains a base class for raising errors when an object's method cannot be
# implemented

__all__ = ['NotImplementedBase']


class NotImplementedBase:
    """Base class for raising errors when an object's method cannot be
    implemented."""
    def _not_implemented(self, func_name: str):
        """Raises a method and object specific not implemented error.

        Parameters
        ----------
        func_name : str
            The name of the method to raise in the NotImplementedError
            exception.
        """
        raise NotImplementedError(f"{func_name} not implemented for "
                                  f"{self.name}")
