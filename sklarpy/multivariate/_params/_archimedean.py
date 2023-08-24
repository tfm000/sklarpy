from sklarpy._other import Params

__all__ = ['MultivariateClaytonParams', 'MultivariateGumbelParams', 'BivariateFrankParams']


class MultivariateArchimedeanParamsBase(Params):
    @property
    def theta(self) -> float:
        return self.to_dict['theta']

    @property
    def d(self) -> int:
        return self.to_dict['d']


class MultivariateClaytonParams(MultivariateArchimedeanParamsBase):
    pass



class MultivariateGumbelParams(MultivariateArchimedeanParamsBase):
    pass


class BivariateFrankParams(MultivariateArchimedeanParamsBase):
    pass
