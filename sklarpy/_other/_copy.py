# Contains a base class used for adding object copying functionality
import copy


__all__ = ['Copyable']


class Copyable:
    """Base class used for adding object copying functionality"""
    def copy(self, name: str = None):
        """Returns a copy of your SklarPy object.

        Parameters
        ----------
        name : str
            The name of your copied object. If None, the name of the original
            is used.

        Returns
        --------
        self:
            Copied object
        """
        if name is None:
            new_name: str = self.name
        elif isinstance(name, str):
            new_name: str = name
        else:
            raise TypeError("name must be a string.")

        copy_object = copy.deepcopy(self)
        copy_object._name = new_name
        return copy_object
