import numpy as np
import scipy.special
from collections import deque
from typing import Tuple
# import scipy.optimize
from scipy.optimize import differential_evolution

from sklarpy.multivariate._prefit_dists import PreFitContinuousMultivariate
from sklarpy._utils import get_iterator
from sklarpy.univariate import gig
from sklarpy.multivariate._distributions._params import MultivariateGenHyperbolicParams
from sklarpy.multivariate import multivariate_normal

__all__ = ['multivariate_gen_hyperbolic']


class multivariate_gen_hyperbolic_gen(PreFitContinuousMultivariate):
    _ASYMMETRIC: bool = True

    def _pdf(self, x: np.ndarray, params: tuple, **kwargs) -> np.ndarray:
        # getting params
        lambda_: float = params[0]
        chi: float = params[1]
        psi: float = params[2]
        loc: np.ndarray = params[3]
        shape: np.ndarray = params[4]
        gamma: np.ndarray = params[5]

        # reshaping for matrix multiplication
        d: int = loc.size
        loc = loc.reshape((d, 1))
        gamma = gamma.reshape((d, 1))
        x = x.reshape((d, 1))

        # common calculations
        shape_inv: np.ndarray = np.linalg.inv(shape)
        q: float = chi + ((x - loc).T @ shape_inv @ (x - loc))
        p: float = psi + (gamma.T @ shape_inv @ gamma)
        r: float = np.sqrt(chi * psi)

        # calculating normalising constant
        c_numerator: float = (r ** - lambda_) * (psi ** lambda_) * (p ** ((d / 2) - lambda_))
        c_denominator: float = ((2 * np.pi) ** (d / 2)) * (np.linalg.det(shape) ** 0.5) * scipy.special.kv(lambda_, r)
        c: float = c_numerator / c_denominator

        # calculating main fraction
        numerator: float = np.exp((x - loc).T @ shape_inv @ gamma) * scipy.special.kv(lambda_ - (d / 2), np.sqrt(q * p))
        denominator: float = np.sqrt(q * p) ** ((d / 2) - lambda_)

        return c * numerator / denominator

    def __singlular_cdf(self, num_variables: int, xrow: np.ndarray, params: tuple) -> float:
        def integrable_pdf(*xrow):
            return self._pdf(xrow, params)

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

    def __w_rvs(self, size: int, params: tuple) -> np.ndarray:
        # getting params
        lambda_: float = params[0]
        chi: float = params[1]
        psi: float = params[2]

        return gig.rvs((size, ), (chi, psi, lambda_))

    def _rvs(self, size: int, params: tuple) -> np.ndarray:
        # getting params
        loc: np.ndarray = params[3]
        shape: np.ndarray = params[4]
        gamma: np.ndarray = params[5]

        # calculating required values
        A: np.ndarray = np.linalg.cholesky(shape)
        num_variables: int = loc.size

        # reshaping for matrix multiplication
        loc = loc.reshape((num_variables, 1))
        gamma = gamma.reshape((num_variables, 1))

        # generating rvs
        z: np.ndarray = np.random.normal(0, 1, (num_variables, size))
        w: np.ndarray = self.__w_rvs(size, params)

        # generating rvs
        m: np.ndarray = loc + (w * gamma)
        return (m + np.sqrt(w) * (A @ z)).T

    @staticmethod
    def _gig_x_a(a: float, lambda_: float, chi: float, psi: float) -> float:
        r: float = np.sqrt(psi * chi)
        return ((chi/psi) ** (a/2)) * (scipy.special.kv(lambda_ + a, r) / scipy.special.kv(lambda_, r))

    def _etas_deltas_zetas(self, data: float, params: tuple, d, calc_zetas: bool = False, h: float = 10 ** -5) -> tuple:
        lambda_, chi, psi, loc, shape, gamma = params
        shape_inv: np.ndarray = np.linalg.inv(shape)
        etas: deque = deque()
        deltas: deque = deque()
        zetas: deque = deque()
        for xi in data:
            xi = xi.reshape(loc.shape)
            cond_lam: float = lambda_ - 0.5*d
            cond_chi: float = float((xi - loc).T @  shape_inv @ (xi - loc) + chi)
            cond_psi: float = float(psi + gamma.T @ shape_inv @ gamma)

            eta_i: float = self._gig_x_a(1, cond_lam, cond_chi, cond_psi)
            delta_i: float = self._gig_x_a(1, - cond_lam, cond_psi, cond_chi)

            if calc_zetas:
                zeta_i: float = (self._gig_x_a(h, cond_lam, cond_chi, cond_psi) - self._gig_x_a(-h, cond_lam, cond_chi, cond_psi))/ (2*h)
                zetas.append(zeta_i)
            etas.append(eta_i)
            deltas.append(delta_i)

        n: int = len(etas)
        if calc_zetas:
            return np.asarray(etas).reshape((n, 1)), np.asarray(deltas).reshape((n, 1)), np.asarray(zetas).reshape((n, 1))
        return np.asarray(etas).reshape((n, 1)), np.asarray(deltas).reshape((n, 1))

    def _q2(self, w_params: np.ndarray, etas: np.ndarray, deltas: np.ndarray, zetas: np.ndarray) -> float:
        # etas: np.ndarray, deltas: np.ndarray, zetas: np.ndarray
        # etas, deltas, zetas = others
        lambda_, chi, psi = w_params
        n: int = zetas.size
        return (lambda_ - 1) * zetas.sum() - 0.5*chi*deltas.sum() - 0.5*psi*etas.sum() + 0.5 * n * lambda_ * np.log(psi/chi) - n * np.log(2*scipy.special.kv(lambda_, np.sqrt(chi*psi)))

    def _fit_given_data(self, data: np.ndarray, **kwargs) -> Tuple[dict, bool]:
        # getting optimization parameters from kwargs
        maxiter: int = kwargs.get('maxiter', 100)
        miniter: int = kwargs.get('miniter', min(10, maxiter))
        tol: float = kwargs.get('tol', 10**-3)
        q2_options = kwargs.get('q2_options', {})
        default_q2_options: dict = {'bounds': ((-10.0, 10.0), (0.01, 10.0), (0.01, 10.0)), 'maxiter': 1000, 'tol': 0.01}
        for arg in default_q2_options:
            if arg not in q2_options:
                q2_options[arg] = default_q2_options[arg]

        # 1. initialization
        k: int = 1
        lambda_: float = -1.0
        chi: float = 0.5
        psi: float = 0.5
        loc: np.ndarray = data.mean(axis=0, dtype=float)
        loc = loc.reshape((loc.size, 1))
        shape: np.ndarray = np.cov(data, rowvar=False, dtype=float)
        gamma: np.ndarray = np.zeros(loc.shape, dtype=float)

        d: int = loc.size
        shape_scale: float = np.linalg.norm(shape)
        x_bar: np.ndarray = data.mean(axis=0, dtype=float)

        q1q2: float = -np.inf
        converged: bool = False
        while (k <= maxiter): # or (not converged):
            # 2. calculate weights
            print(k, '.',2)
            etas, deltas = self._etas_deltas_zetas(data, (lambda_, chi, psi, loc, shape, gamma), d)
            eta_mean, delta_mean = etas.mean(), deltas.mean()

            # 3. update gamma
            print(k,'.', 3)
            if self._ASYMMETRIC:
                gamma = ((deltas * (x_bar - data)).mean(axis=0, dtype=float) / (delta_mean * eta_mean - 1)).reshape((d, 1))

            # 4. update location vector and dispersion matrix
            print(k,'.', 4)
            loc = ((deltas * data).mean(axis=0, dtype=float).reshape((d, 1)) - gamma) / delta_mean
            omega: np.ndarray = np.array([deltas[i] * (data[i, :] - loc) @ (data[i, :] - loc).T for i in range(data.shape[0])]).mean(axis=0, dtype=float) - (eta_mean*gamma@gamma.T)
            shape = ((shape_scale / np.linalg.norm(omega)) ** (1 / d)) * omega

            # 5. recalculate weights
            print(k,'.', 5)
            etas, deltas, zetas = self._etas_deltas_zetas(data, (lambda_, chi, psi, loc, shape, gamma), d, calc_zetas=True)

            # 6. maximise Q2
            print(k, '.',6)

            q2_res = differential_evolution(self._q2, args=(etas, deltas, zetas), **q2_options)
            lambda_, chi, psi = q2_res['x']

            # 7. check convergence
            print(k, '.', 7)
            exp_w: float = self._gig_x_a(1, lambda_, chi, psi)

            q1_val: float = multivariate_normal.loglikelihood(data, (loc + exp_w*gamma, exp_w*shape))
            old_q1q2, q1q2 = q1q2, q1_val + q2_res['fun']
            change: float = q1q2 - old_q1q2
            converged = True if q2_res['success'] and change >= 0.0 and change <= tol and k >= miniter else False
            k += 1
            print(k, ':', q1q2)
        breakpoint()
        return {'lambda_': lambda_, 'chi': chi, 'psi': psi, 'loc': loc, 'shape': shape, 'gamma': gamma}, converged

    def _fit_copula(self, data: np.ndarray, **kwargs) -> Tuple[dict, bool]:
        raise NotImplemented("not yet implemented. TODO")

    def _fit_given_params_tuple(self, params: tuple, **kwargs) -> Tuple[dict, int]:
        # getting kwargs
        raise_cov_error: bool = kwargs.get('raise_cov_error', True)
        raise_corr_error: bool = kwargs.get('raise_corr_error', False)

        # checking correct number of parameters
        super()._fit_given_params_tuple(params)

        # checking valid loc vector and shape matrix
        lambda_, chi, psi, loc, shape, gamma = params
        self._check_loc_shape(loc, shape, check_shape_valid_cov=raise_cov_error, check_shape_valid_corr=raise_corr_error)

        # checking valid gamma vector
        num_variables: int = loc.size
        if not isinstance(gamma, np.ndarray) or gamma.size != num_variables:
            raise ValueError("gamma vector must be a numpy array with the same size as the location vector.")

        # checking lambda, chi and psi
        for i, param in enumerate(params[:3]):
            if not (isinstance(param, float) or isinstance(param, int)):
                raise TypeError("lambda, chi and psi must be integers or floats.")
            if i > 0:
                if param <= 0:
                    raise ValueError("chi and psi must be strictly positive integers or floats.")

        return {'lambda_': lambda_, 'chi': chi, 'psi': psi, 'loc': loc, 'shape': shape, 'gamma': gamma}, num_variables


multivariate_gen_hyperbolic: multivariate_gen_hyperbolic_gen = multivariate_gen_hyperbolic_gen(name="multivariate_gen_hyperbolic", params_obj=MultivariateGenHyperbolicParams, num_params=6, max_num_variables=np.inf)


if __name__ == '__main__':
    my_loc = np.array([1, -3], dtype=float)
    my_shape = np.array([[1, 0.7], [0.7, 1]], dtype=float)
    my_gamma = np.array([2.3, 1.4], dtype=float)
    my_lambda = - 5
    my_chi = 1.7
    my_psi = 4.5
    my_params = (my_lambda, my_chi, my_psi, my_loc, my_shape, my_gamma)

    # rvs = multivariate_gen_hyperbolic.rvs(10000, my_params)
    import pandas as pd
    # df = pd.DataFrame(rvs)
    # df.to_excel('test_data.xlsx')
    df = pd.read_excel('test_data.xlsx', index_col=0)
    rvs = df.to_numpy()

    my_genhyp = multivariate_gen_hyperbolic.fit(rvs)
