from sklarpy._other import Params

__all__ = ['MultivariateGaussianKDEParams']


class MultivariateGaussianKDEParams(Params):
    @property
    def kde(self):
        return self.to_dict['kde']
