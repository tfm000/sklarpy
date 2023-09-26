# Contains code for evaluating Modified Bessel functions
import numpy as np
import scipy.special
from typing import Union, Iterable

__all__ = ['kv']


class kv:
    """Modified Bessel function of the 2nd kind."""
    __SMALL_VALUE: float = 10 ** -5
    __LARGE_VALUE: float = 100

    @staticmethod
    def logkv(v: Union[float, int], z: Union[float, int], **kwargs) -> float:
        """Evaluates the log of the Modified Bessel function of the 2nd kind.
        Accounts for the limits of z and v.

        Parameters
        -----------
        v: Union[float, int]
            The v parameter, which specifies the member of the Modified Bessel
            function of the 2nd kind family to evaluate.
        z : Union[int, float]
            The scalar value to evaluate the Modified Bessel function of the
            2nd kind at.
        kwargs:
            See below

        Keyword Arguments
        ------------------
        small_value: float
            The value of which, if z is below, we assume K_v(z) tends to the
            limit of K_v(0).
            Default is 10 ** -5
        large_value: float
            The value of which, if z is above, we assume K_v(z) tends to the
            limit of K_v(inf).
            Default is 100

        Returns
        -------
        logkv: float
            The value of log(K_v(z))
        """
        # argchecks
        small_value: float = kwargs.get('small_value', kv.__SMALL_VALUE)
        large_value: float = kwargs.get('large_value', kv.__LARGE_VALUE)

        for arg in (v, z, large_value, small_value):
            if isinstance(arg, Iterable) and np.asarray(arg).size == 1:
                arg = float(arg)

            if not (isinstance(arg, float) or isinstance(arg, int)):
                raise TypeError("all arguments / keyword arguments must "
                                "be scalars.")
        # k_-v(z) = k_v(z)
        v = abs(v)

        if 0 <= z <= small_value:
            if v == 0:
                # lim z-> 0, v -> 0
                return np.log(-np.log(z))
            # lim z -> 0, v fixed
            return np.log(0.5) + scipy.special.loggamma(v) - v * np.log(0.5*z)
        elif z >= large_value:
            # lim z -> inf
            return -z + 0.5 * np.log(np.pi / (2*z))

        # no limiting cases
        return np.log(scipy.special.kv(v, z))

    @staticmethod
    def kv(v: float, z: float, **kwargs) -> float:
        """Evaluates the Modified Bessel function of the 2nd kind.
        Accounts for the limits of z and v.

        Parameters
        -----------
        v: Union[float, int]
            The v parameter, which specifies the member of the Modified Bessel
            function of the 2nd kind family to evaluate.
        z : Union[int, float]
            The scalar value to evaluate the Modified Bessel function of the
            2nd kind at.
        kwargs:
            See below

        Keyword Arguments
        ------------------
        small_value: float
            The value of which, if z is below, we assume K_v(z) tends to the
            limit of K_v(0).
            Default is 10 ** -5
        large_value: float
            The value of which, if z is above, we assume K_v(z) tends to the
            limit of K_v(inf).
            Default is 100

        Returns
        -------
        kv: float
            The value of K_v(z)
        """
        return np.exp(kv.logkv(v, z, **kwargs))
