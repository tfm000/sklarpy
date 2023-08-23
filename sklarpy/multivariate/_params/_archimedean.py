from sklarpy._other import Params

__all__ = ['MultivariateClaytonParams', 'MultivariateGumbelParams']


class MultivariateArchimedeanParamsBase(Params):
    @property
    def theta(self) -> float:
        return self.to_dict['theta']


class MultivariateClaytonParams(MultivariateArchimedeanParamsBase):
    @property
    def d(self) -> int:
        return self.to_dict['d']


class MultivariateGumbelParams(MultivariateArchimedeanParamsBase):
    @property
    def d(self) -> int:
        return self.to_dict['d']
