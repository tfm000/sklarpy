from sklarpy._other._serialize import Savable

__all__ = ['Params']


class Params(Savable):
    def __init__(self, params: dict, name: str, num_variables: int = None):
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

    def __getitem__(self, param_name: str):
        if self.__contains__(param_name):
            return self.to_dict[param_name]
        raise KeyError(f"{param_name} not valid")

    def __str__(self):
        return f"{self.name.title()}Params"

    def __repr__(self):
        return self.__str__()

    @property
    def to_dict(self) -> dict:
        return self._params.copy()

    @property
    def to_tuple(self) -> tuple:
        return tuple(self.to_dict.values())

    @property
    def to_list(self) -> list:
        return list(self.to_dict.values())

    @property
    def num_variables(self) -> int:
        return self._num_variables
