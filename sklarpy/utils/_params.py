# Contains a base class for all distribution parameter objects
from typing import Union

from sklarpy.utils._serialize import Savable

__all__ = ['Params']


class Params(Savable):
    """Base class for all distribution parameter objects."""
    def __init__(self, params: dict, name: str, num_variables: int = None):
        """Base class for all distribution parameter objects.

        Parameters
        ----------
        params : dict
            A dictionary with the names of the parameters as keys and their
            scalar / vector / function values as values.
        name : str
            The name of the parameters object.
        num_variables: int
            The number of variables fitted to the distribution.
            Optional. Default is None.
        """
        Savable.__init__(self, name)
        self._params: dict = params
        self._num_variables: int = num_variables

    def __iter__(self) -> iter:
        for param_value in self.to_dict.values():
            yield param_value

    def __len__(self) -> int:
        return len(self.to_dict)

    def __contains__(self, param_name: str) -> bool:
        return param_name in self.to_dict

    def __getitem__(self, index: Union[str, int]):
        """Allows the user to retrieve a parameter value either via its name or
        index value."""
        if self.__contains__(index):
            return self.to_dict[index]
        elif isinstance(index, int) and ((0 <= index < len(self)) or
                                         (-len(self) <= index < 0)):
            return self.to_tuple[index]
        raise KeyError(f"{index} not valid")

    def __str__(self):
        return f"{self.name.title()}Params"

    def __repr__(self):
        return self.__str__()

    @property
    def to_dict(self) -> dict:
        """Transforms the parameters object into a dictionary."""
        return self._params.copy()

    @property
    def to_tuple(self) -> tuple:
        """Transforms the parameters object into a tuple."""
        return tuple(self.to_dict.values())

    @property
    def to_list(self) -> list:
        """Transforms the parameters object into a list."""
        return list(self.to_dict.values())

    @property
    def num_variables(self) -> Union[int, None]:
        """Returns the number of variables fitted to the model.
        """
        return self._num_variables
