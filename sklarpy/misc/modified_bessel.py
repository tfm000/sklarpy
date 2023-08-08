import numpy as np
import scipy.special

__all__ = ['kv']


class kv:
    """Modified Bessel Function of the 2nd Kind"""
    __SMALL_VALUE: float = 10**-5
    __LARGE_VALUE: float = 100

    @staticmethod
    def logkv(v: float, z: float, **kwargs) -> float:
        small_value: float = kwargs.get('small_value', kv.__SMALL_VALUE)
        large_value: float = kwargs.get('large_value', kv.__LARGE_VALUE)
        v = abs(v)

        if 0 <= z <= small_value:
            if v == 0:
                return np.log(-np.log(z))
            return np.log(0.5) + scipy.special.loggamma(v) - v * np.log(0.5*z)
        elif z >= large_value:
            return -z + 0.5 * np.log(np.pi / (2*z))
        return np.log(scipy.special.kv(v, z))

    @staticmethod
    def kv(v: float, z: float, **kwargs) -> float:
        return np.exp(kv.logkv(v, z, **kwargs))
