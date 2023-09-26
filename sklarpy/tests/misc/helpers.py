# Contains helper functions for testing SklarPy misc code
import numpy as np
from typing import Union

__all__ = ['XCubed', 'Exp', 'Log']


class DifferentiableBase:
    @staticmethod
    def f(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """The function to evaluate.

        Parameters
        ----------
        x: Union[float, np.ndarray]
            Values to evaluate the function at.

        Returns
        -------
        fx: Union[float, np.ndarray]
            The function evaluated at the given values.
        """

    @staticmethod
    def dfdx(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """The first derivative of the function.

        Parameters
        ----------
        x: Union[float, np.ndarray]
            Values to evaluate the derivative at.

        Returns
        -------
        dfdx: Union[float, np.ndarray]
            The derivative evaluated at the given values.
        """


class XCubed(DifferentiableBase):
    @staticmethod
    def f(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return np.power(x, 3)

    @staticmethod
    def dfdx(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return 3 * np.power(x, 2)


class Exp(DifferentiableBase):
    @staticmethod
    def f(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return np.exp(x)

    @staticmethod
    def dfdx(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return np.exp(x)


class Log(DifferentiableBase):
    @staticmethod
    def f(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return np.log(x)

    @staticmethod
    def dfdx(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return np.power(x, -1)
