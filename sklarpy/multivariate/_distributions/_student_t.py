# contains code for the multivariate student t distribution
import numpy as np
import scipy.stats
import scipy.integrate
import scipy.optimize
from collections import deque
from typing import Tuple

from sklarpy.multivariate._prefit_dists import PreFitContinuousMultivariate
from sklarpy._utils import get_iterator
from sklarpy.misc import CorrelationMatrix
from sklarpy.multivariate._distributions._params import MultivariateStudentTParams

__all__ = ['multivariate_student_t']


class multivariate_student_t_gen(PreFitContinuousMultivariate):
    def _pdf(self, x: np.ndarray, params: tuple, **kwargs) -> np.ndarray:
        return scipy.stats.multivariate_t.pdf(x, loc=params[0].flatten(), shape=params[1], df=params[2])

    def __singlular_cdf(self, num_variables: int, xrow: np.ndarray, params: tuple) -> float:
        def integrable_pdf(*xrow):
            return scipy.stats.multivariate_t.pdf(xrow, loc=params[0].flatten(), shape=params[1], df=params[2])

        ranges = [[-np.inf, float(xrow[i])] for i in range(num_variables)]
        res: tuple = scipy.integrate.nquad(integrable_pdf, ranges)
        return res[0]

    def _cdf(self, x: np.ndarray, params: tuple, **kwargs) -> np.ndarray:
        num_variables: int = x.shape[1]

        show_progress: bool = kwargs.get('show_progress', True)
        iterator = get_iterator(x, show_progress, "calculating cdf values")

        cdf_values: deque = deque()
        for xrow in iterator:
            val: float = self.__singlular_cdf(num_variables, xrow, params)
            cdf_values.append(val)
        return np.asarray(cdf_values)

    def _rvs(self, size: int, params: tuple) -> np.ndarray:
        return scipy.stats.multivariate_t.rvs(size=size, loc=params[0].flatten(), shape=params[1], df=params[2])

    def __cov_to_shape(self, cov: np.ndarray, dof: float) -> np.ndarray:
        return cov * (dof - 2) / dof

    def _objective_func(self, dof: float, mu: np.ndarray, shape: np.ndarray, data: np.ndarray, shape_is_cov: bool):
        if shape_is_cov:
            shape: np.ndarray = self.__cov_to_shape(shape, dof)
        return - self.loglikelihood(data, (mu, shape, dof))

    def __fit_dof(self, mu: np.ndarray, shape: np.ndarray, shape_is_cov: bool, data: np.ndarray, **kwargs) -> Tuple[float, bool]:
        dof_bounds: tuple = kwargs.pop('dof_bounds', (2.01, 100.0))

        # checking dof bounds
        if len(dof_bounds) != 2:
            raise ValueError("dof_bounds must contain exactly 2 elements")

        new_dof_bounds = deque()
        eps = 10 ** -9
        for val in dof_bounds:
            if val < 0:
                raise ValueError("dof bounds must all be strictly positive.")
            if val == 0:
                val = eps
            new_dof_bounds.append(val)
        new_dof_bounds = new_dof_bounds
        lb, ub = min(new_dof_bounds), max(new_dof_bounds)
        if lb == ub:
            if lb > eps / 2:
                lb -= eps / 2
                ub += eps / 2
            else:
                ub += eps

        # finding dof
        res = scipy.optimize.differential_evolution(self._objective_func, [(lb, ub)], args=(mu, shape, data, shape_is_cov))
        dof: float = float(res['x'])
        success: bool = res['success']
        return dof, success

    def _fit_given_data(self, data: np.ndarray, **kwargs) -> Tuple[dict, bool]:
        method: str = kwargs.pop('method', 'laloux_pp_kendall')

        # finding covariance and mean arrays (shape != cov)
        mu: np.ndarray = data.mean(axis=0, dtype=float)
        cov: np.ndarray = CorrelationMatrix(data).cov(method=method, **kwargs)

        # finding dof
        dof, success = self.__fit_dof(mu=mu, shape=cov, shape_is_cov=True, data=data)

        # calculating shape parameter
        shape: np.ndarray = self.__cov_to_shape(cov, dof)
        return {'mu': mu, 'shape': shape, 'dof': dof}, success

    def _fit_copula(self, data: np.ndarray, **kwargs) -> Tuple[dict, bool]:
        method: str = kwargs.pop('method', 'laloux_pp_kendall')

        # finding covariance and mean arrays (shape != cov)
        mu: np.ndarray = np.zeros((data.shape[1],), dtype=float)
        corr: np.ndarray = CorrelationMatrix(data).corr(method=method, **kwargs)

        # finding dof
        dof, success = self.__fit_dof(mu=mu, shape=corr, shape_is_cov=False, data=data)

        return {'mu': mu, 'shape': corr, 'dof': dof}, success

    def _fit_given_params_tuple(self, params: tuple, **kwargs) -> Tuple[dict, int]:
        # getting kwargs
        raise_cov_error: bool = kwargs.get('raise_cov_error', True)
        raise_corr_error: bool = kwargs.get('raise_corr_error', False)

        # checking correct number of parameters
        super()._fit_given_params_tuple(params)

        # checking valid mean vector and shape matrix
        mu, shape, dof = params
        self._check_loc_shape(mu, shape, check_shape_valid_cov=raise_cov_error, check_shape_valid_corr=raise_corr_error)

        # checking valid degrees of freedom parameter
        if (not (isinstance(dof, float) or isinstance(dof, int))) or (dof <= 0):
            raise TypeError("dof must be a positive float or integer.")

        return {'mu': mu, 'shape': shape, 'dof': dof}, mu.size


multivariate_student_t: multivariate_student_t_gen = multivariate_student_t_gen(name="multivariate_student_t", params_obj=MultivariateStudentTParams, num_params=3, max_num_variables=np.inf)


if __name__ == "__main__":
    my_mu = np.array([1, -3], dtype=float)
    my_corr = np.array([[1, 0.7], [0.7, 1]], dtype=float)
    my_sig = np.array([1.3, 2.5])
    my_dof = 3.4

    my_cov = np.diag(my_sig) @ my_corr @ np.diag(my_sig)
    my_shape = my_cov * (my_dof - 2) / my_dof

    rvs = multivariate_student_t.rvs(1000, (my_mu, my_shape, my_dof))
    my_mv_t = multivariate_student_t.fit(rvs)
    my_mv_t.mc_cdf_plot()

    # t2 = multivariate_student_t.fit(params=my_mv_t.params)
    # print(t2.mc_cdf(np.array([[1, -3]])))
