import numpy as np
# import scipy.special
from collections import deque
from typing import Tuple, Union
from scipy.optimize import differential_evolution

from sklarpy.multivariate._prefit_dists import PreFitContinuousMultivariate
from sklarpy.univariate import gig
from sklarpy.misc import CorrelationMatrix, kv

__all__ = ['multivariate_gen_hyperbolic_gen']


class multivariate_gen_hyperbolic_gen(PreFitContinuousMultivariate):
    _ASYMMETRIC: bool = True
    _DATA_FIT_METHODS = (*PreFitContinuousMultivariate._DATA_FIT_METHODS, 'em')
    _NUM_W_PARAMS: int = 3

    def _check_w_params(self, params: tuple) -> None:
        # checking lambda, chi and psi
        for i, param in enumerate(params[:3]):
            if not (isinstance(param, float) or isinstance(param, int)):
                raise TypeError("lambda, chi and psi must be integers or floats.")
            if i > 0:
                if param <= 0:
                    raise ValueError("chi and psi must be strictly positive integers or floats.")

    def _check_params(self, params: tuple, **kwargs) -> None:
        # checking correct number of params passed
        super()._check_params(params)

        # checking lambda, chi and psi
        self._check_w_params(params)

        # checking valid location vector and shape matrix
        loc, shape, gamma = params[3:]
        definiteness, ones = kwargs.get('definiteness', 'pd'), kwargs.get('ones', False)
        self._check_loc_shape(loc, shape, definiteness, ones)

        # checking valid gamma vector
        num_variables: int = loc.size
        if not isinstance(gamma, np.ndarray) or gamma.size != num_variables:
            raise ValueError("gamma vector must be a numpy array with the same size as the location vector.")

    def _singular_logpdf(self, x: np.ndarray, params: tuple, **kwargs) -> float:
        # getting params
        lamb, chi, psi, loc, shape, gamma = params

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

        log_c: float = (lamb * (np.log(psi) - np.log(r))) + ((0.5*d - lamb) * np.log(p)) - 0.5 * ((d*np.log(2*np.pi)) + np.log(np.linalg.det(shape)) + 2*kv.logkv(lamb, r))
        log_h: float = kv.logkv(lamb - (d / 2), np.sqrt(q * p)) + ((x - loc).T @ shape_inv @ gamma) - (0.25 * (d - 2 * lamb) * (np.log(q) + np.log(p)))
        return float(log_c + log_h)

    def _logpdf(self, x: np.ndarray, params: tuple, **kwargs) -> np.ndarray:
        xshape: tuple = x.shape
        if len(xshape) == 1:
            x.reshape((x.shape[0], 1))
        elif len(xshape) != 2:
            raise ValueError("x must be a 1 or 2 dimensional array")

        return np.array([self._singular_logpdf(xrow, params, **kwargs) for xrow in x], dtype=float)

    def _w_rvs(self, size: int, params: tuple) -> np.ndarray:
        return gig.rvs((size,), params[:3], ppf_approx=True)

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
        w: np.ndarray = self._w_rvs(size, params)

        # generating rvs
        m: np.ndarray = loc + (w * gamma)
        return (m + np.sqrt(w) * (A @ z)).T

    def _etas_deltas_zetas(self, data: np.ndarray, params: tuple, h: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        lamb, chi, psi, loc, shape, gamma = params
        shape_inv: np.ndarray = np.linalg.inv(shape)
        d: int = loc.size
        etas: deque = deque()
        deltas: deque = deque()
        zetas: deque = deque()

        p: float = psi + (gamma.T @ shape_inv @ gamma)
        v = lamb - 0.5 * d

        for xi in data:
            xi = xi.reshape((d, 1))
            qi: float = chi + ((xi - loc).T @ shape_inv @ (xi - loc))

            cond_params: tuple = (v, qi, p)
            eta_i: float = self._exp_w(cond_params)
            delta_i: float = self._exp_w((-v, p, qi))
            zeta_i: float = self._exp_log_w(cond_params, h)

            deltas.append(delta_i)
            etas.append(eta_i)
            zetas.append(zeta_i)

        n: int = len(etas)
        return np.asarray(etas).reshape((n, 1)), np.asarray(deltas).reshape((n, 1)), np.asarray(zetas).reshape((n, 1))

    def _neg_q2(self, w_params: np.ndarray, etas: np.ndarray, deltas: np.ndarray, zetas: np.ndarray) -> float:
        lamb, chi, psi = w_params
        n: int = zetas.size
        a: float = n * (0.5 * lamb * np.log(psi / chi) - np.log(2) - kv.logkv(lamb, (chi * psi) ** 0.5))
        q2: float = a + ((lamb - 1) * zetas.sum()) + (-0.5 * chi * deltas.sum()) + (-0.5 * psi * etas.sum())
        return -q2

    def _gh_to_params(self, params: tuple) -> tuple:
        return params

    def _em(self, data: np.ndarray, min_retries: int, max_retries: int, copula: bool, bounds: tuple, params0: Union[np.ndarray, None], cov_method: str, miniter: int, maxiter: int, h: float, tol: float, q2_options: dict, randomness_var: float, convergence_window_length: int, show_progress: bool, **kwargs) -> Tuple[tuple, bool]:
        min_retries = max(0, min_retries)
        max_retries = max(min_retries, 1, max_retries)
        convergence_window_length = max(1, convergence_window_length)
        miniter = max(convergence_window_length, miniter)
        maxiter = max(miniter, maxiter)

        # values constant between runs
        n, d = data.shape

        # initializing parameters
        shape0: np.ndarray = CorrelationMatrix(data).corr(method=cov_method, **kwargs) if copula else CorrelationMatrix(data).cov(method=cov_method)
        shape0_eigenvalues: np.ndarray = np.linalg.eigvals(shape0)
        S_det: float = shape0_eigenvalues.prod()

        # other constants
        min_eig: float = shape0_eigenvalues.min()  # used to keep shape matrix positive definite
        x_bar: np.ndarray = data.mean(axis=0, dtype=float)

        if params0 is not None:
            # if theta0 is fixed, function outcome is ~ deterministic between runs
            max_retries = 1

        # initializing run
        rerun: bool = True
        run: int = 0
        runs_loglikelihoods, runs_params, successful_runs = deque(), deque(), deque()
        while rerun:
            # getting initial starting parameters
            if params0 is None:
                # generating new theta 0
                theta0: np.ndarray = self._get_low_dim_theta0(data=data, bounds=bounds, copula=copula)
                p0: tuple = self._get_params(params=self._low_dim_theta_to_params(theta=theta0, S=shape0, S_det=S_det, min_eig=min_eig, copula=copula))
            else:
                p0: np.ndarray = params0.copy()

            # doing a single expectation-maximisation run
            params, success, k, loglikelihood = self._em_single_run(data=data, copula=copula, miniter=miniter, maxiter=maxiter, h=h, tol=tol, q2_options=q2_options, randomness_var=randomness_var, convergence_window_length=convergence_window_length, show_progress=show_progress, bounds=bounds, n=n, d=d, params0=p0, min_eig=min_eig, shape_scale=S_det, x_bar=x_bar, run=run)

            # storing results
            runs_params.append(params)
            runs_loglikelihoods.append(loglikelihood)
            if success:
                successful_runs.append(run)

            # checking whether to do another run
            if run + 1 >= max_retries:
                rerun = False
            elif run >= min_retries:
                rerun = ((not success) or np.isinf(loglikelihood) or np.isnan(loglikelihood))
            run += 1

        # determining best run and whether convergence was achieved
        runs_loglikelihoods = np.asarray(runs_loglikelihoods, dtype=float)
        if len(successful_runs) > 0:
            # at least one EM run converged
            best_run: int = successful_runs[np.argmax(runs_loglikelihoods[successful_runs])]
            converged: bool = True
        else:
            # convergence not achieved
            best_run: int = np.argmax(runs_loglikelihoods)
            converged: bool = False

        if show_progress:
            print(f"EM Optimisation Complete. Converged= {converged}, f(x)= {-runs_loglikelihoods[best_run]}")
        return self._gh_to_params(runs_params[best_run]), converged

    def _add_randomness(self, params: tuple, bounds: tuple, d: int, randomness_var: float, copula: bool) -> tuple:
        adj_params: deque = deque()
        for i, param in enumerate(params):
            if not (i == 4 or (i==3 and copula)):
                eps: float = np.random.normal(0, randomness_var)
                adj_param = param + eps * param

                if i <= 2:
                    # scalar parameters
                    param_bounds: tuple = bounds[i]
                    adj_param = min(max(adj_param, param_bounds[0]), param_bounds[1])
                else:
                    # vector parameters
                    param_bounds: np.ndarray = np.array(bounds[3:][:d]) if i == 3 else np.array(bounds[-d:])
                    adj_param = np.concatenate([adj_param, param_bounds[:, 0].reshape(d, 1)], axis=1).max(axis=1).reshape(d, 1)
                    adj_param = np.concatenate([adj_param, param_bounds[:, 1].reshape(d, 1)], axis=1).min(axis=1).reshape(d, 1)

            else:
                # shape matrix or location vector when copula
                adj_param = param
            adj_params.append(adj_param)
        return tuple(adj_params)

    def _q2_opt(self, bounds: tuple, etas: np.ndarray, deltas: np.ndarray, zetas: np.ndarray, q2_options: dict):
        return differential_evolution(self._neg_q2, bounds=bounds[:3], args=(etas, deltas, zetas), **q2_options)

    def _em_single_run(self, data: np.ndarray, copula: bool, miniter: int, maxiter: int, h: float, tol: float, q2_options: dict, randomness_var: float, convergence_window_length: int, show_progress: bool, bounds: tuple, n: int, d: int, params0: tuple, min_eig: float, shape_scale: float, x_bar: np.ndarray, run: int) -> Tuple[tuple, bool]:
        # getting optimization parameters from kwargs

        # 1. initialization
        k: int = 1
        best_k: int = 0
        continue_em: bool = True
        best_params = params = (lamb, chi, psi, loc, shape, gamma) = params0
        best_loglikelihood = loglikelihood = self.loglikelihood(data, best_params)
        q2_success: bool = False
        last_m_runs: deque = deque([best_loglikelihood])  # m = convergence_window_length

        while (k<miniter) or (k <= maxiter and continue_em):
            if show_progress:
                print(f"EM run {run+1}, step {k-1}: q2 converged= {q2_success}, f(x)= {-loglikelihood}")

            if np.isinf(loglikelihood) or np.isnan(loglikelihood):
                # reuse start using best params
                adj_params: tuple = self._add_randomness(best_params, bounds, d, randomness_var, copula)
                params = (lamb, chi, psi, loc, shape, gamma) = adj_params

            # 2. Calculate Weights
            etas, deltas, _ = self._etas_deltas_zetas(data, params, h)
            eta_mean, delta_mean = etas.mean(), deltas.mean()

            # 3. Update gamma
            if self._ASYMMETRIC:
                gamma = ((deltas * (x_bar - data)).mean(axis=0, dtype=float) / (delta_mean * eta_mean - 1)).reshape((d, 1))

            if not copula:
                # 4. update location and shape
                loc = ((deltas * data).mean(axis=0, dtype=float).reshape((d, 1)) - gamma) / delta_mean
                omega: np.ndarray = np.array([deltas[i] * (data[i, :] - loc) @ (data[i, :] - loc).T for i in range(n)]).mean(axis=0, dtype=float) - (eta_mean * gamma @ gamma.T)
                omega, _, eigenvalues = CorrelationMatrix._rm_pd(omega, min_eig)
                shape = ((shape_scale / eigenvalues.prod()) ** (1 / d)) * omega

            # 5. recalculate weights
            etas, deltas, zetas = self._etas_deltas_zetas(data, (lamb, chi, psi, loc, shape, gamma), h)

            # 6. maximise Q2
            q2_res = self._q2_opt(bounds=bounds, etas=etas, deltas=deltas, zetas=zetas, q2_options=q2_options)
            lamb, chi, psi = q2_res['x']
            q2_success: bool = q2_res['success']

            # 7. check convergence
            params: tuple = (lamb, chi, psi, loc, shape, gamma)
            loglikelihood: float = self.loglikelihood(data, params)

            if q2_success and k > convergence_window_length:
                max_change: float = abs(np.asarray(last_m_runs, dtype=float) - loglikelihood).max()
                continue_em = max_change > tol

                last_m_runs.append(loglikelihood)
                last_m_runs.popleft()
            else:
                last_m_runs.append(loglikelihood)

            if q2_success and (loglikelihood >= best_loglikelihood):
                best_loglikelihood = loglikelihood
                best_params = params
                best_k = k

            k += 1

        max_change = abs(np.asarray(last_m_runs, dtype=float) - best_loglikelihood).max()
        converged: bool = max_change <= tol

        return best_params, converged, best_k, best_loglikelihood

    @staticmethod
    def _theta_to_params(theta: np.ndarray, d: int) -> tuple:
        lamb, chi, psi = theta[:3]
        loc: np.ndarray = theta[3: d+3].copy()
        shape: np.ndarray = PreFitContinuousMultivariate._shape_from_array(theta[d+3: -d], d)
        gamma: np.ndarray = theta[-d:].copy()
        return lamb, chi, psi, loc, shape, gamma

    @staticmethod
    def _exp_w_a(params: tuple, a: float) -> float:
        lamb, chi, psi = params[:3]
        r: float = np.sqrt(chi * psi)
        if r > 100:
            # tends to 1 as r -> inf
            bessel_val: float = 1.0
        else:
            bessel_val: float = kv.kv(lamb+a, r) / kv.kv(lamb, r)
            # bessel_val: float = scipy.special.kv(lamb+a, r) / scipy.special.kv(lamb, r)
        return ((chi/psi) ** (a/2)) * bessel_val

    @staticmethod
    def _exp_w(params: tuple) -> float:
        return multivariate_gen_hyperbolic_gen._exp_w_a(params, 1)

    @staticmethod
    def _var_w(params: tuple) -> float:
        return multivariate_gen_hyperbolic_gen._exp_w_a(params, 2) - (multivariate_gen_hyperbolic_gen._exp_w_a(params, 1) ** 2)

    @staticmethod
    def _exp_log_w(params: tuple, h: float) -> float:
        return (multivariate_gen_hyperbolic_gen._exp_w_a(params, h) - multivariate_gen_hyperbolic_gen._exp_w_a(params, -h)) / 2*h

    def _get_bounds(self, data: np.ndarray, as_tuple: bool, **kwargs) -> Union[dict, tuple]:
        # calculating default bounds
        d: int = data.shape[1]
        data_bounds: np.ndarray = np.array([data.min(axis=0), data.max(axis=0)], dtype=float).T
        data_abs_max: np.ndarray = abs(data).max(axis=0)
        data_extremes: np.ndarray = np.array([-data_abs_max, data_abs_max], dtype=float).T
        default_bounds: dict = {'lamb': (-10.0, 10.0), 'chi': (0.01, 10.0), 'psi': (0.01, 10.0), 'loc': data_bounds, 'gamma': data_extremes}
        return super()._get_bounds(default_bounds, d, as_tuple, **kwargs)

    def _get_low_dim_theta0(self, data: np.ndarray, bounds: tuple, copula: bool) -> np.ndarray:
        d: int = data.shape[1]
        lamb0: float = np.random.uniform(*bounds[0])
        chi0: float = np.random.uniform(*bounds[1])
        psi0: float = np.random.uniform(*bounds[2])
        gamma0: np.ndarray = np.zeros((d,), dtype=float)
        if not copula:
            loc0: np.ndarray = data.mean(axis=0, dtype=float).flatten()
            return np.array([lamb0, chi0, psi0, *loc0, *gamma0], dtype=float)
        return np.array([lamb0, chi0, psi0, *gamma0], dtype=float)

    def _low_dim_theta_to_params(self, theta: np.ndarray, S: np.ndarray, S_det: float, min_eig: float, copula: bool) -> tuple:
        d: int = S.shape[0]

        lamb, chi, psi = theta[:3]
        gamma: np.ndarray = theta[-d:].copy()
        gamma = gamma.reshape((d, 1))

        if not copula:
            # getting location vector from theta
            loc: np.ndarray = theta[3: d + 3].copy()
            loc = loc.reshape((d, 1))

            # calculating implied shape parameter
            exp_w: float = self._exp_w(theta[: 3])
            var_w: float = self._var_w(theta[: 3])
            omega: np.ndarray = (S - var_w * gamma @ gamma.T) / exp_w
            omega, _, eigenvalues = CorrelationMatrix._rm_pd(omega, min_eig)  # ensuring pd
            shape: np.ndarray = omega * ((S_det / eigenvalues.prod()) ** (1/d))  # solving identifiability problem
        else:
            loc: np.ndarray = np.zeros((d, 1), dtype=float)
            shape: np.ndarray = S
        return lamb, chi, psi, loc, shape, gamma

    def _fit_given_data_kwargs(self, method: str, data: np.ndarray, **user_kwargs) -> dict:
        if method == 'em':
            bounds: tuple = self._get_bounds(data, True, **user_kwargs)
            default_param0 = None
            default_q2_options: dict = {'maxiter': 1000, 'tol': 0.01}
            q2_options: dict = user_kwargs.get('q2_options', {})
            for arg in default_q2_options:
                if arg not in q2_options:
                    q2_options[arg] = default_q2_options[arg]
            kwargs: dict = {'min_retries': 0, 'max_retries': 3, 'copula': False, 'bounds': bounds, 'params0': default_param0, 'cov_method': 'laloux_pp_kendall', 'miniter': 10, 'maxiter': 100, 'h': 10**-5, 'tol': 0.1, 'q2_options': q2_options, 'randomness_var': 0.1, 'convergence_window_length': 5, 'show_progress': False}
        else:
            kwargs: dict = super()._fit_given_data_kwargs(method, data, **user_kwargs)
        return kwargs

    def _fit_given_params_tuple(self, params: tuple, **kwargs) -> Tuple[dict, int]:
        self._check_params(params, **kwargs)
        return {'lamb': params[0], 'chi': params[1], 'psi': params[2], 'loc': params[3], 'shape': params[4], 'gamma': params[5]}, params[3].size

    def num_scalar_params(self, d: int, copula: bool = False, **kwargs) -> int:
        vec_num: int = d if copula else 2 * d
        vec_num -= 0 if self._ASYMMETRIC else d
        return self._NUM_W_PARAMS + vec_num + self._num_shape_scalar_params(d=d, copula=copula)


# if __name__ == '__main__':
#     my_loc = np.array([1, -3, 5.2], dtype=float)
#     my_shape = np.array([[1, 0.284, 0.520], [0.284, 1, 0.435], [0.520, 0.435, 1]], dtype=float)
#     my_gamma = np.array([2.3, 1.4, -4.3], dtype=float)
#     my_lambda = - 0.5
#     my_chi = 1.7
#     my_psi = 4.5
#     my_params = (my_lambda, my_chi, my_psi, my_loc, my_shape, my_gamma)
#
#     # theta = np.array([my_lambda, my_chi, my_psi, *my_loc.tolist(), *my_shape.diagonal().tolist(), *my_shape[0,1:].tolist(), my_shape[1, 2], *my_gamma.tolist()], dtype=float)
#     # breakpoint()
#     # testparams = multivariate_gen_hyperbolic.theta_to_params(theta, 3)
#     # breakpoint()
#     # breakpoint()
#     # gigrvs = gig.rvs((1000, 1), (my_lambda, my_chi, my_psi), ppf_approx=True)
#     # # gigrvs = gig.rvs((1000, 1), (my_chi, my_psi, my_lambda), ppf_approx=True)
#     # fgig = gig.fit(gigrvs)
#     # fgig.plot()
#     # fgig.rvs((100000, 1), ppf_approx=True)
#     # breakpoint()
#
#     #
#     rvs = multivariate_gen_hyperbolic.rvs(10000, my_params)
#
#     # multivariate_gen_hyperbolic.mc_cdf_plot(params=my_params)
#     # print(rvs)
#     # import pandas as pd
#     # pd.DataFrame(rvs).to_excel('gh_rvs.xlsx')
#     # test_run = multivariate_gen_hyperbolic._low_dim_mle(rvs)
#     #
#     pdf = multivariate_gen_hyperbolic.pdf(rvs, my_params)
#     # print(pdf)
#     # pdf = multivariate_gen_hyperbolic._singular_pdf(rvs[k, :], my_params)
#     # print(np.exp(log_pdf) - pdf)
#
#     # df = pd.DataFrame(rvs)
#     # df.to_excel('test_data2.xlsx')
#     # df = pd.read_excel('test_data2.xlsx', index_col=0)
#     # rvs = df.to_numpy()
#
#     # max_loglikelihood = multivariate_gen_hyperbolic.loglikelihood(rvs, my_params)
#     # print(f"target: {max_loglikelihood}")
#     # print(multivariate_gen_hyperbolic.cdf(rvs[0, :], my_params))
#     # print(multivariate_gen_hyperbolic.cdf(rvs[:6, :], my_params))
#
#     # my_genhyp = multivariate_gen_hyperbolic.fit(rvs, show_progress=True, copula=True)
#     # print(my_genhyp.params.to_dict)
#     # print(my_genhyp.loglikelihood())
#
#     my_genhyp = multivariate_gen_hyperbolic.fit(rvs, method='em', show_progress=True, min_retries=1, max_retries=1, maxiter=50)
#     print(my_genhyp.params.to_dict)
#     print(my_genhyp.loglikelihood())
#
#     print('theoretical max: ', multivariate_gen_hyperbolic.loglikelihood(rvs, my_params))
#     # my_genhyp.pdf_plot()
#
#     # multivariate_gen_hyperbolic.pdf_plot(params=my_params)
#
#     # import matplotlib.pyplot as plt
#     # fig = plt.figure()
#     # ax = fig.add_subplot(projection='3d')
#     # pdf_vals = multivariate_gen_hyperbolic.pdf(rvs, my_params)
#     # ax.scatter(rvs[:, 0], rvs[:, 1], pdf_vals)
#     # multivariate_gen_hyperbolic.pdf_plot(params=my_params, show=False)
#     # plt.show()
#
