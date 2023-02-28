import pandas as pd
import scipy.interpolate
import numpy as np
from collections import deque
from typing import Callable, Union

from sklarpy.copulas.numerical._numerical import Numerical
from sklarpy.copulas.numerical._numerical_params import EmpiricalCopulaParams

__all__ = ['EmpiricalCopula']


class EmpiricalCopula(Numerical):
    _OBJ_NAME = "EmpiricalCopula"
    _MAX_NUM_VARIABLES = np.inf
    _PARAMS_OBJ = EmpiricalCopulaParams

    def _get_interpolator_kwargs(self, interpolator: str, kwargs: dict) -> dict:
        if interpolator == 'LinearND':
            return {}

        num_data_points: int = self._marginals.shape[0]
        rbf_defaults: dict = {'neighbors': min(100, num_data_points), 'smoothing': 1.0}
        for arg, val in rbf_defaults.items():
            if arg not in kwargs:
                kwargs[arg] = val
        return kwargs

    def fit(self, interpolator: str = 'RBF', params: EmpiricalCopulaParams = None, eps: float = 0.01, delta: float = 0.5,
            num_additional_points: int = 10 ** 4, **kwargs):
        """Fits the EmpiricalCopula.

        Parameters
        =============
        interpolator: str
            The name of the interpolator to use when fitting the empirical pdf function. Note that LinearNDInterpolator
            is always used to fit the cdf function, as this is less noisy in general.
                - 'RBF' for RBFInterpolator
                - 'LinearND' for LinearNDInterpolator.
            Default is 'RBF'
        params: EmpiricalCopulaParams
            The parameters object of a previously fitted EmpiricalCopula.
        eps: float
            The radius of the 'ball', |p - x| <= eps, around points, p, to use when fitting the empirical pdf function.
            If you have a small number of data points, you may want to increase this value.
            Default is 0.01
        num_additional_points: int
            The number of additional random uniformly distributed points to generate, where the pdf and cdf functions
            are also fitted at, increasing the accuracy of interpolating functions.
            Default is 10**4.
        kwargs: dict
            The arguments to pass to the interpolator function when fitting the pdf.
                For RBFInterpolator:
                    - neighbors
                    - smoothing
                    - kernel
                    - epsilon
                    - degree
                For LinearNDInterpolator:
                    - None
            See more below.

        Keyword arguments
        =================
        The arguments to pass to the interpolator function when fitting the pdf.

        neighbors: For RBFInterpolator.
            The number of neighbors to use in when calculating the value of the interpolant.
            Default is min(100, num_data_points). Pass None to use all data points (not recommended when number of
            data points exceeds 1000, as can be resource intensive).
        smoothing: For RBFInterpolator.
            The smoothing parameter. The interpolant perfectly fits the data when this is
            set to 0. For large values, the interpolant approaches a least squares fit of a polynomial with the
            specified degree. Default is 1.0
        kernel: For RBFInterpolator.
            The type of RBF. This should be one of
                - 'linear' : -r
                - 'thin_plate_spline' : r**2 * log(r)
                - 'cubic' : r**3
                - 'quintic' : -r**5
                - 'multiquadric' : -sqrt(1 + r**2)
                - 'inverse_multiquadric' : 1/sqrt(1 + r**2)
                - 'inverse_quadratic' : 1/(1 + r**2)
                - 'gaussian' : exp(-r**2)
            Default is 'thin_plate_spline'.
        epsilon: For RBFInterpolator.
            Shape parameter that scales the input to the RBF. If kernel is 'linear', 'thin_plate_spline', 'cubic', or
            'quintic', this defaults to 1 and can be ignored because it has the same effect as scaling the smoothing
            parameter. Otherwise, this must be specified.
        degree: For RBFInterpolator.
            Degree of the added polynomial. For some RBFs the interpolant may not be well-posed if the polynomial degree
            is too small. Those RBFs and their corresponding minimum degrees are
                - 'multiquadric' : 0
                - 'linear' : 0
                - 'thin_plate_spline' : 1
                - 'cubic' : 1
                - 'quintic' : 2
            The default value is the minimum degree for kernel or 0 if there is no minimum degree. Set this to -1 for no added polynomial.

        Returns
        ========
        self:
            self
        """
        self._fitting = True

        if not self._fit_params(params):
            # User did not provide params

            # checks
            if not isinstance(interpolator, str):
                raise TypeError("interpolator must be a string.")
            elif interpolator not in ("RBF", "LinearND"):
                raise ValueError("interpolator must be 'RBF' or 'LinearND'")

            for float_arg in (eps, delta):
                if not (isinstance(float_arg, float) and (float_arg > 0)):
                    raise ValueError("eps and delta must be positive float values.")

            if not (isinstance(num_additional_points, int) and (num_additional_points >= 0)):
                raise ValueError("num_additional_points must be a positive integer or zero.")

            if self._marginals is None:
                raise ValueError("if params are not specified, marginal cdf values must be provided in init.")

            shape: tuple = self._marginals.shape
            self._check_num_variables(shape[1])

            # generating additional data points
            self._umins, self._umaxs = self._marginals.min(axis=0), self._marginals.max(axis=0)
            additional_points: np.ndarray = np.random.uniform(self._umins - delta, self._umaxs + delta,
                                                              (num_additional_points, self._num_variables))
            empirical_range: np.ndarray = np.concatenate([
                self._marginals,
                additional_points,
                (self._umins - delta).reshape((1, self._num_variables)),
                (self._umaxs + delta).reshape((1, self._num_variables)),
            ], axis=0)

            # getting pdf and cdf values
            num_data_points: int = shape[0]
            non_zero_points = deque()
            pdf_vals: Union[deque, np.ndarray] = deque()
            cdf_vals: Union[deque, np.ndarray] = deque()
            for i, row in enumerate(empirical_range):
                cdf_val: float = np.all(self._marginals <= row, axis=1).sum() / num_data_points
                pdf_val: float = np.all((self._marginals >= row - eps) & (self._marginals <= row + eps),
                                        axis=1).sum() / num_data_points

                cdf_vals.append(cdf_val)
                pdf_vals.append(pdf_val)
                if pdf_val != 0.0:
                    non_zero_points.append(i)
            cdf_vals = np.asarray(cdf_vals)
            pdf_vals = np.asarray(pdf_vals)

            # fitting pdf and cdf functions using interpolation
            pdf_kwargs: dict = self._get_interpolator_kwargs(interpolator, kwargs)
            pdf_interpolator: Callable = eval(f'scipy.interpolate.{interpolator}Interpolator')
            self._pdf_func = pdf_interpolator(empirical_range[non_zero_points, :], pdf_vals[non_zero_points],
                                              **pdf_kwargs)
            self._cdf_func = scipy.interpolate.LinearNDInterpolator(empirical_range, cdf_vals)

        self._params = {'pdf_func': self._pdf_func, 'cdf_func': self._cdf_func, 'umins': self._umins,
                        'umaxs': self._umaxs, 'num_variables': self._num_variables}
        self._fitting = False
        self._fitted = True

        return self


if __name__ == "__main__":
    from sklarpy import load

    std6 = load('std6')
    rvs = std6.rvs(10000)
    rvs: pd.DataFrame = pd.DataFrame(rvs, columns=['weather', 'ice cream consumed'])
    emp = EmpiricalCopula(rvs, 'ice cream & weather')
    emp.fit()
    # std6.pdf_plot()
    emp.pdf_plot()
    emp.cdf_plot()
    emp.marginal_pairplot(alpha=0.1)