# Contains code for the multivariate Generalized Hyperbolic model
import numpy as np
import pandas as pd
from collections import deque
from typing import Tuple, Union
from scipy.optimize import differential_evolution

from sklarpy.multivariate._prefit_dists import PreFitContinuousMultivariate
from sklarpy.multivariate._fitted_dists import FittedContinuousMultivariate
from sklarpy.univariate import gig
from sklarpy.univariate._distributions import _gh
from sklarpy.misc import CorrelationMatrix, kv
from sklarpy.utils._params import Params

__all__ = ['multivariate_gen_hyperbolic_gen']


class multivariate_gen_hyperbolic_gen(PreFitContinuousMultivariate):
    """Multivariate Generalized Hyperbolic model."""
    _ASYMMETRIC: bool = True
    _DATA_FIT_METHODS = ('mle', 'em')
    _NUM_W_PARAMS: int = 3
    _UNIVAR = _gh

    def _check_w_params(self, params: tuple) -> None:
        """Performs checks the parameters of the positive mixing variable W
        and raises an error if one or more is not valid.

        Parameters
        ----------
        params : tuple
            The parameters which define the multivariate model, in tuple form.
        """
        # checking lambda, chi and psi
        for i, param in enumerate(params[:3]):
            if not (isinstance(param, float) or isinstance(param, int)):
                raise TypeError("lambda, chi and psi must be scalars.")
            if i > 0:
                if param <= 0:
                    raise ValueError("chi and psi must be strictly positive "
                                     "scalar values.")

    def _check_params(self, params: tuple, **kwargs) -> None:
        # checking correct number of params passed
        super()._check_params(params)

        # checking lambda, chi and psi
        self._check_w_params(params)

        # checking valid location vector and shape matrix
        loc, shape, gamma = params[3:]
        definiteness: str = kwargs.get('definiteness', 'pd')
        ones: bool = kwargs.get('ones', False)
        self._check_loc_shape(loc, shape, definiteness, ones)

        # checking valid gamma vector
        num_variables: int = loc.size
        if not isinstance(gamma, np.ndarray) or gamma.size != num_variables:
            raise ValueError("gamma vector must be a numpy array with the "
                             "same size as the location vector.")

    def _get_dim(self, params: tuple) -> int:
        return params[3].size

    def _singular_logpdf(self, xrow: np.ndarray, params: tuple, **kwargs) \
            -> float:
        """Returns the log-pdf value for a single set of variable observations.

        Parameters
        ----------
        xrow : np.ndarray
            A single set of variable observations.
        params : tuple
            The parameters which define the multivariate model, in tuple form.

        Returns
        -------
        single_logpdf : float
            The log-pdf value for a single set of variable observations.
        """

        # getting params
        lamb, chi, psi, loc, shape, gamma = params

        # reshaping for matrix multiplication
        d: int = loc.size
        loc = loc.reshape((d, 1))
        gamma = gamma.reshape((d, 1))
        xrow = xrow.reshape((d, 1))

        # common calculations
        shape_inv: np.ndarray = np.linalg.inv(shape)
        q: float = chi + ((xrow - loc).T @ shape_inv @ (xrow - loc))
        p: float = psi + (gamma.T @ shape_inv @ gamma)
        r: float = np.sqrt(chi * psi)

        log_c: float = (lamb * (np.log(psi) - np.log(r))) \
                       + ((0.5 * d - lamb) * np.log(p)) \
                       - 0.5 * (
                               (d * np.log(2 * np.pi))
                               + np.log(np.linalg.det(shape))
                               + 2 * kv.logkv(lamb, r)
                       )
        log_h: float = kv.logkv(lamb - (d / 2), np.sqrt(q * p)) \
                       + ((xrow - loc).T @ shape_inv @ gamma) \
                       - (0.25 * (d - 2 * lamb) * (np.log(q) + np.log(p)))
        return float(log_c + log_h)

    def _logpdf(self, x: np.ndarray, params: tuple, **kwargs) -> np.ndarray:
        xshape: tuple = x.shape
        if len(xshape) == 1:
            x.reshape((x.shape[0], 1))
        elif len(xshape) != 2:
            raise ValueError("x must be a 1 or 2 dimensional array")

        return np.array([self._singular_logpdf(xrow, params, **kwargs)
                         for xrow in x], dtype=float)

    def _w_rvs(self, size: int, params: tuple) -> np.ndarray:
        """Returns random variates, generated from the univariate distribution
        of W.

        Parameters
        ----------
        size: int
            How many multivariate random samples to generate from the
            multivariate distribution.
        params : tuple
            The parameters which define the multivariate model, in tuple form.

        See Also
        --------
        McNeil Frey and Embrechts:
            3.2 Normal Mixture Distributions.

        Returns
        -------
        w_rvs: np.ndarray
            univariate array of random variables, sampled from distribution W.
        """
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

    def _etas_deltas_zetas(self, data: np.ndarray, params: tuple, h: float) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculates the conditional expectations of W given X, to use in the
        EM algorithm.

        Parameters
        ----------
        data: np.ndarray
            An array of multivariate data to fit parameters using EM algorithm.
        params : tuple
            The parameters which define the multivariate model, in tuple form.
        h: float
            When fitting to data only.
            The h parameter to use in numerical differentiation in the 'em'
            algorithm.
            Default value is 10 ** -5

        See Also
        --------
        McNeil Frey and Embrechts:
            equations 3.37

        Returns
        -------
        etas, deltas, zetas: Tuple[np.ndarray, np.ndarray, np.ndarray]
            array of eta values, array of delta values, array of zeta values.
        """
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
            eta_i: float = self._UNIVAR._exp_w(cond_params)
            delta_i: float = self._UNIVAR._exp_w((-v, p, qi))
            zeta_i: float = self._exp_log_w(cond_params, h)

            deltas.append(delta_i)
            etas.append(eta_i)
            zetas.append(zeta_i)

        n: int = len(etas)
        return (
            np.asarray(etas, dtype=float).reshape((n, 1)),
            np.asarray(deltas, dtype=float).reshape((n, 1)),
            np.asarray(zetas, dtype=float).reshape((n, 1))
        )

    def _neg_q2(self, w_params: np.ndarray, etas: np.ndarray,
                deltas: np.ndarray, zetas: np.ndarray) -> float:
        """Calculates the negative of the Q2 function described by McNeil,
        Frey and Embrechts, to use as an objective function when optimizing in
        the EM algorithm.

        Parameters
        ----------
        w_params : np.ndarray
            lamb, chi, psi in a numpy array.
        etas: np.ndarray
            array of eta values.
        deltas: np.ndarray
            array of delta values.
        zetas: np.ndarray
            array of zeta values.

        See Also
        --------
        McNeil Frey and Embrechts:
            equations 3.38

        Returns
        -------
        neg_q2: float
            The negative of the q2 function described by McNeil, Frey and
            Embrechts.
        """
        lamb, chi, psi = w_params
        n: int = zetas.size
        a: float = n * (
                0.5 * lamb * np.log(psi / chi)
                - np.log(2)
                - kv.logkv(lamb, (chi * psi) ** 0.5)
        )
        q2: float = a + ((lamb - 1) * zetas.sum()) \
                    + (-0.5 * chi * deltas.sum()) \
                    + (-0.5 * psi * etas.sum())
        return -q2

    def _gh_to_params(self, params: tuple) -> tuple:
        """Converts parameters from Generalized Hyperbolic form, into
        distribution specific tuples. Used for special cases of the
        Generalized Hyperbolic distribution.

        Parameters
        ----------
        params : tuple
            The parameters which define the Generalized Hyperbolic
            multivariate model, in tuple form.

        Returns
        -------
        params : tuple
            The parameters which define the multivariate model, in tuple form.
        """
        return params

    def _em(self, data: np.ndarray, min_retries: int, max_retries: int,
            copula: bool, bounds: tuple, params0: Union[tuple, None],
            cov_method: str, miniter: int, maxiter: int, h: float, tol: float,
            min_eig: Union[float, None], q2_options: dict,
            randomness_var: float, convergence_window_length: int,
            show_progress: bool, **kwargs) -> Tuple[tuple, bool]:
        """Performs a modified version of the Expectation-Maximization (EM)
        algorithm outlined by McNeil, Frey and Embrechts.

        As McNeil, Frey and Embrechts noted, their EM algorithm occasionally
        may result in NAN or inf log-likelihoods, for certain parameter values.
        When these occur, our modification takes our prior best parameters,
        adds some random noise, before continuing.

        Parameters
        ----------
        data: np.ndarray
            An array of multivariate data to fit parameters using EM algorithm.
        min_retries: int
            The minimum number of times to re-run the EM algorithm if
            convergence is not achieved. If params0 are given, each retry will
            continue to use this as a starting point. If params0 not given, a
            new set of random values for each parameter will be generated
            before each run.
        max_retries: int
            The maximum number of times to re-run the EM algorithm if
            convergence is not achieved. If params0 are given, each retry will
            continue to use this as a starting point. If params0 not given, a
            new set of random values for each parameter will be generated
            before each run.
        copula: bool
            True if the distribution is a copula distribution.
        bounds: tuple
            The bounds to use in parameter fitting / optimization, as a tuple.
        params0: Union[tuple, None]
            An initial estimate of the parameters to use when starting the
            optimization algorithm. If params0 fixed, function outcome ~
            deterministic between runs.
        cov_method: str
            The method to use when estimating the sample covariance matrix.
            See CorrelationMatrix.cov for more information.
        miniter: int
            The minimum number of iterations to perform for each EM run,
            regardless of whether convergence has been achieved.
        maxiter: int
            The maximum number of iterations to perform for each EM run,
            regardless of whether convergence has been achieved.
        h: float
            The h parameter to use in numerical differentiation in the 'em'
            algorithm.
        tol: float
            The tolerance to use when determine convergence. For the 'em'
            algorithm, convergence is achieved when the optimization to find
            lambda, chi and psi (by maximizing Q2) successfully converges + the
            log-likelihood for the current run is our greatest observation +
            the change in log-likelihood over a given window length of runs
            is <= tol.
        min_eig: Union[float, None]
            The delta / smallest positive eigenvalue to allow when enforcing
            positive definiteness via Rousseeuw and Molenberghs' technique.
        q2_options: dict
            A dictionary of keyword arguments to pass to scipy's
            differential_evolution non-convex solver, when maximising Q2.
        randomness_var: float
            A scalar value of the variance of random noise to add to
            parameters.
        convergence_window_length: int
            The window length to use when determining convergence in the 'em'
            algorithm. Convergence is achieved when the optimization to find
            lambda, chi and psi (by maximizing Q2) successfully converges + the
            log-likelihood for the current run is our greatest observation +
            the change in log-likelihood over a given window length of runs
            is <= tol.
        show_progress: bool
            True to display the progress of the EM algorithm.
        kwargs:
            Keyword arguments to pass to the CorrelationMatrix.cov

        See Also
        --------
        McNeil Frey and Embrechts:
            Algorithm 3.14
        CorrelationMatrix.cov
        scipy.optimize.differential_evolution

        Returns
        -------
        res: Tuple[tuple, bool]
            The parameters optimized to fit the data,
            True if convergence was successful false otherwise.
        """
        min_retries = max(0, min_retries)
        max_retries = max(min_retries, 1, max_retries)
        convergence_window_length = max(1, convergence_window_length)
        miniter = max(convergence_window_length, miniter)
        maxiter = max(miniter, maxiter)

        # values constant between runs
        n, d = data.shape

        # initializing parameters
        shape0: np.ndarray = CorrelationMatrix(data).corr(
            method=cov_method, **kwargs) if copula \
            else CorrelationMatrix(data).cov(method=cov_method)
        shape0_eigenvalues: np.ndarray = np.linalg.eigvals(shape0)
        S_det: float = shape0_eigenvalues.prod()

        # other constants
        min_eig: float = shape0_eigenvalues.min() if min_eig is None \
            else min_eig  # used to shape matrix pd
        x_bar: np.ndarray = data.mean(axis=0, dtype=float)

        if params0 is not None:
            # if params0 fixed, function outcome ~ deterministic between runs
            max_retries = 1

        # initializing run
        rerun: bool = True
        run: int = 0
        runs_loglikelihoods, runs_params, successful_runs = (
            deque(), deque(), deque())
        while rerun:
            # getting initial starting parameters
            p0: tuple = self._get_params(
                self._get_params0(
                    data=data, bounds=bounds, copula=copula,
                    cov_method=cov_method, min_eig=min_eig, em_opt=True,
                    **kwargs)
                if params0 is None else params0)

            # doing a single expectation-maximisation run
            params, success, k, loglikelihood = self._em_single_run(
                data=data, copula=copula, miniter=miniter, maxiter=maxiter,
                h=h, tol=tol, q2_options=q2_options,
                randomness_var=randomness_var,
                convergence_window_length=convergence_window_length,
                show_progress=show_progress, bounds=bounds, n=n, d=d,
                params0=p0, min_eig=min_eig, shape_scale=S_det, x_bar=x_bar,
                run=run)

            # storing results
            runs_params.append(params)
            runs_loglikelihoods.append(loglikelihood)
            if success:
                successful_runs.append(run)

            # checking whether to do another run
            if run + 1 >= max_retries:
                rerun = False
            elif run >= min_retries:
                rerun = ((not success)
                         or np.isinf(loglikelihood)
                         or np.isnan(loglikelihood))
            run += 1

        # determining best run and whether convergence was achieved
        runs_loglikelihoods = np.asarray(runs_loglikelihoods, dtype=float)
        if len(successful_runs) > 0:
            # at least one EM run converged
            best_run: int = successful_runs[np.argmax(
                runs_loglikelihoods[successful_runs])]
            converged: bool = True
        else:
            # convergence not achieved
            best_run: int = np.argmax(runs_loglikelihoods)
            converged: bool = False

        if show_progress:
            print(f"EM Optimisation Complete. Converged= {converged}, "
                  f"f(x)= {-runs_loglikelihoods[best_run]}")
        return self._gh_to_params(runs_params[best_run]), converged

    def _add_randomness(self, params: tuple, bounds: tuple, d: int,
                        randomness_var: float, copula: bool) -> tuple:
        """Modification to the EM algorithm proposed by McNeil, Frey and
        Embrechts. Adds random noise to a given set of parameters.

        Parameters
        ----------
        params : tuple
            The parameters which define the multivariate model, in tuple form.
        bounds: tuple
            The bounds to use in parameter fitting / optimization, as a tuple.
        randomness_var: float
            A scalar value of the variance of random noise to add to
            parameters.
        copula: bool
            True if the distribution is a copula distribution.

        Returns
        -------
        adj_params : tuple
            A tuple of params with some added noise.
        """
        adj_params: deque = deque()
        for i, param in enumerate(params):
            if not (i == 4 or (i == 3 and copula)):
                size = 1 if i <= 2 else (d, 1)
                eps: float = np.random.normal(0, randomness_var, size)
                adj_param = param + eps * param

                if i <= 2:
                    # scalar parameters
                    param_bounds: tuple = bounds[i]
                    adj_param = min(max(float(adj_param), param_bounds[0]),
                                    param_bounds[1])
                else:
                    # vector parameters
                    param_bounds: np.ndarray = np.array(bounds[3:][:d]) \
                        if i == 3 else np.array(bounds[-d:])
                    adj_param = np.concatenate([
                        adj_param, param_bounds[:, 0].reshape(d, 1)
                    ], axis=1).max(axis=1).reshape(d, 1)
                    adj_param = np.concatenate([
                        adj_param, param_bounds[:, 1].reshape(d, 1)
                    ], axis=1).min(axis=1).reshape(d, 1)

            else:
                # shape matrix or location vector when copula
                adj_param = param
            adj_params.append(adj_param)
        return tuple(adj_params)

    def _q2_opt(self, bounds: tuple, etas: np.ndarray, deltas: np.ndarray,
                zetas: np.ndarray, q2_options: dict):
        """Maximizes the Q2 function described by McNeil, Frey and Embrechts
        using scipy's differential_evolution non-convex optimizer.

        Parameters
        ----------
        bounds: tuple
            The bounds to use in parameter fitting / optimization, as a tuple.
        etas: np.ndarray
            array of eta values.
        deltas: np.ndarray
            array of delta values.
        zetas: np.ndarray
            array of zeta values.
        q2_options: dict
            A dictionary of keyword arguments to pass to scipy's
            differential_evolution non-convex solver, when maximising Q2.

        See Also
        --------
        McNeil Frey and Embrechts:
            equations 3.38

        Returns
        -------
        res:
            The results of the optimization.
        """
        return differential_evolution(self._neg_q2, bounds=bounds[:3],
                                      args=(etas, deltas, zetas), **q2_options)

    def _em_single_run(self, data: np.ndarray, copula: bool, miniter: int,
                       maxiter: int, h: float, tol: float, q2_options: dict,
                       randomness_var: float, convergence_window_length: int,
                       show_progress: bool, bounds: tuple, n: int, d: int,
                       params0: tuple, min_eig: float, shape_scale: float,
                       x_bar: np.ndarray, run: int
                       ) -> Tuple[tuple, bool, int, float]:
        """Performs a single run of the modified version of the
        Expectation-Maximization (EM) algorithm outlined by McNeil, Frey and
        Embrechts.

        As McNeil, Frey and Embrechts noted, their EM algorithm occasionally
        may result in NAN or inf log-likelihoods, for certain parameter values.
        When these occur, our modification takes our prior best parameters,
        adds some random noise, before continuing.

        Parameters
        ----------
        data: np.ndarray
            An array of multivariate data to fit parameters using EM algorithm.
        copula: bool
            True if the distribution is a copula distribution.
        miniter: int
            The minimum number of iterations to perform for each EM run,
            regardless of whether convergence has been achieved.
        maxiter: int
            The maximum number of iterations to perform for each EM run,
            regardless of whether convergence has been achieved.
        h: float
            The h parameter to use in numerical differentiation in the 'em'
            algorithm.
        tol: float
            The tolerance to use when determine convergence. For the 'em'
            algorithm, convergence is achieved when the optimization to find
            lambda, chi and psi (by maximizing Q2) successfully converges + the
            log-likelihood for the current run is our greatest observation +
            the change in log-likelihood over a given window length of runs
            is <= tol.
        q2_options: dict
            A dictionary of keyword arguments to pass to scipy's
            differential_evolution non-convex solver, when maximising Q2.
        randomness_var: float
            A scalar value of the variance of random noise to add to
            parameters.
        convergence_window_length: int
            The window length to use when determining convergence in the 'em'
            algorithm. Convergence is achieved when the optimization to find
            lambda, chi and psi (by maximizing Q2) successfully converges + the
            log-likelihood for the current run is our greatest observation +
            the change in log-likelihood over a given window length of runs
            is <= tol.
        show_progress: bool
            True to display the progress of the EM algorithm.
        bounds: tuple
            The bounds to use in parameter fitting / optimization, as a tuple.
        n: int
            The number of observations / data rows
        d: int
            The dimension / number of variables.
        params0: tuple
            An initial estimate of the parameters to use when starting the
            optimization algorithm, as a tuple.
        min_eig: float
            The delta / smallest positive eigenvalue to allow when enforcing
            positive definiteness via Rousseeuw and Molenberghs' technique.
        shape_scale: float
            The value to scale the shape matrix by each iteration. Solves the
            identifyability problem identified by McNeil, Frey and Embrechts.
        x_bar: np.ndarray
            numpy array containing the means of each variable.
        run: int
            The EM algorithm run number. Used when displaying progress.

        See Also
        --------
        McNeil Frey and Embrechts:
            Algorithm 3.14
        CorrelationMatrix.cov
        scipy.optimize.differential_evolution

        Returns
        -------
        res: Tuple[tuple, bool, int, float]
            The optimal params, True if converged, best iteration,
            optimal loglikelihood
        """

        # getting optimization parameters from kwargs

        # 1. initialization
        k: int = 1
        best_k: int = 0
        continue_em: bool = True
        best_params = params = (lamb, chi, psi, loc, shape, gamma) = params0
        loglikelihood: float = self.loglikelihood(data, best_params)
        best_loglikelihood: float = loglikelihood
        q2_success: bool = False
        last_m_runs: deque = deque([best_loglikelihood])  # m = window length

        while (k < miniter) or (k <= maxiter and continue_em):
            if show_progress:
                print(f"EM run {run+1}, step {k-1}: q2 converged= "
                      f"{q2_success}, f(x)= {-loglikelihood}")

            if np.isinf(loglikelihood) or np.isnan(loglikelihood):
                # reuse start using best params
                adj_params: tuple = self._add_randomness(
                    params=best_params, bounds=bounds, d=d,
                    randomness_var=randomness_var, copula=copula)
                params = (lamb, chi, psi, loc, shape, gamma) = adj_params

            # 2. Calculate Weights
            etas, deltas, _ = self._etas_deltas_zetas(data, params, h)
            eta_mean, delta_mean = etas.mean(), deltas.mean()

            # 3. Update gamma
            if self._ASYMMETRIC:
                gamma = ((deltas * (x_bar - data)).mean(axis=0, dtype=float) /
                         (delta_mean * eta_mean - 1)).reshape((d, 1))

            if not copula:
                # 4. update location and shape
                loc = ((deltas * data).mean(axis=0, dtype=float)
                       .reshape((d, 1)) - gamma) / delta_mean
                omega: np.ndarray = np.array([
                    deltas[i] * (data[i, :] - loc) @ (data[i, :] - loc).T
                    for i in range(n)]).mean(axis=0, dtype=float) - (
                        eta_mean * gamma @ gamma.T)
                omega, _, eigenvalues = CorrelationMatrix._rm_pd(omega,
                                                                 min_eig)
                shape = ((shape_scale / eigenvalues.prod()) ** (1 / d)) * omega

            # 5. recalculate weights
            etas, deltas, zetas = self._etas_deltas_zetas(
                data, (lamb, chi, psi, loc, shape, gamma), h)

            # 6. maximise Q2
            q2_res = self._q2_opt(bounds=bounds, etas=etas, deltas=deltas,
                                  zetas=zetas, q2_options=q2_options)
            lamb, chi, psi = q2_res['x']
            q2_success: bool = q2_res['success']

            # 7. check convergence
            params: tuple = (lamb, chi, psi, loc, shape, gamma)
            loglikelihood: float = self.loglikelihood(data, params)

            if q2_success and k > convergence_window_length:
                max_change: float = abs(
                    np.asarray(last_m_runs, dtype=float) - loglikelihood
                ).max()
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

        max_change = abs(
            np.asarray(last_m_runs, dtype=float) - best_loglikelihood
        ).max()
        converged: bool = max_change <= tol

        return best_params, converged, best_k, best_loglikelihood

    @staticmethod
    def _exp_log_w(params: tuple, h: float) -> float:
        """Calculates the expectation of the log of the distribution W,
        E[log(W)].

        Parameters
        ----------
        params : tuple
            The parameters which define the multivariate model, in tuple form.

        Returns
        -------
        exp_log_w: float
            E[log(W)]
        """
        return (multivariate_gen_hyperbolic_gen._UNIVAR._exp_w_a(params, h)
                - multivariate_gen_hyperbolic_gen._UNIVAR._exp_w_a(params, -h)
                ) / (2 * h)

    def _get_bounds(self, data: np.ndarray, as_tuple: bool, **kwargs
                    ) -> Union[dict, tuple]:
        # calculating default bounds
        d: int = data.shape[1]
        data_abs_max: np.ndarray = abs(data).max(axis=0)
        data_extremes: np.ndarray = np.array([-data_abs_max, data_abs_max],
                                             dtype=float).T
        default_bounds: dict = {'lamb': (-10.0, 10.0), 'chi': (0.01, 10.0),
                                'psi': (0.01, 10.0), 'gamma': data_extremes}
        return super()._get_bounds(default_bounds, d, as_tuple, **kwargs)

    def _theta_to_params(self, theta: np.ndarray, mean: np.ndarray,
                         S: np.ndarray, S_det: float, min_eig: float,
                         copula: bool, **kwargs) -> tuple:
        # extracting lambda, chi, psi and gamma parameters from theta
        d: int = theta.size - 3

        lamb, chi, psi = theta[:3]
        gamma: np.ndarray = theta[3:].copy().reshape((d, 1))

        if copula:
            loc: np.ndarray = np.zeros((d, 1), dtype=float)
            shape: np.ndarray = S
        else:
            # getting central moments of W
            exp_w: float = self._UNIVAR._exp_w(theta[:3])
            var_w: float = self._UNIVAR._var_w(theta[:3])

            # calculating implied location parameter
            loc: np.ndarray = mean - (exp_w * gamma)

            # calculating implied shape parameter
            omega: np.ndarray = (S - var_w * gamma @ gamma.T) / exp_w
            omega, _, eigenvalues = CorrelationMatrix._rm_pd(omega, min_eig)
            shape: np.ndarray = omega * ((S_det / eigenvalues.prod()) ** (1/d))

        return self._gh_to_params(params=(lamb, chi, psi, loc, shape, gamma))

    def _params_to_theta(self, params: tuple, **kwargs) -> np.ndarray:
        return np.array([*params[:3], *params[-1].flatten()], dtype=float)

    def _get_mle_objective_func_kwargs(self, data: np.ndarray, cov_method: str,
                                     min_eig: float, copula: bool, **kwargs
                                     ) -> dict:
        # for GH dists, we return the args to pass to _theta_to_params
        kwargs: dict = {'copula': copula}
        d: int = data.shape[1]

        if copula:
            kwargs['mean'] = np.zeros((d, 1))
            S: np.ndarray = CorrelationMatrix(data).corr(method=cov_method,
                                                        **kwargs)
        else:
            kwargs['mean'] = data.mean(axis=0).reshape((d, 1))
            S: np.ndarray = CorrelationMatrix(data).cov(method=cov_method,
                                                        **kwargs)
        kwargs['S'] = S
        kwargs['S_det'] = np.linalg.det(S)
        if min_eig is None:
            eigenvalues: np.ndarray = np.linalg.eigvals(S)
            min_eig: float = eigenvalues.min()
        kwargs['min_eig'] = min_eig
        return kwargs

    def _get_params0(self, data: np.ndarray, bounds: tuple, cov_method: str,
                     min_eig: float, copula: bool, **kwargs) -> tuple:
        # getting theta0
        d: int = data.shape[1]
        lamb0: float = np.random.uniform(*bounds[0])
        chi0: float = np.random.uniform(*bounds[1])
        psi0: float = np.random.uniform(*bounds[2])
        gamma0: np.ndarray = np.zeros((d,), dtype=float)
        theta0: float = self._params_to_theta(
            params=(lamb0, chi0, psi0, gamma0), **kwargs)

        # converting to params0
        mle_kwargs: dict = self._get_mle_objective_func_kwargs(
            data=data, cov_method=cov_method, min_eig=min_eig,
            copula=copula, **kwargs)
        return self._theta_to_params(theta0, **mle_kwargs)

    def _fit_given_data_kwargs(self, method: str, data: np.ndarray,
                               **user_kwargs) -> dict:
        if method == 'em':
            bounds: tuple = self._get_bounds(data, True, **user_kwargs)
            default_param0 = None
            default_q2_options: dict = {'maxiter': 1000, 'tol': 0.01}
            q2_options: dict = user_kwargs.get('q2_options', {})
            for arg in default_q2_options:
                if arg not in q2_options:
                    q2_options[arg] = default_q2_options[arg]
            kwargs: dict = {
                'min_retries': 0, 'max_retries': 3, 'copula': False,
                'bounds': bounds, 'params0': default_param0,
                'cov_method': 'laloux_pp_kendall', 'miniter': 10,
                'maxiter': 100, 'h': 10**-5, 'tol': 0.1, 'min_eig': None,
                'q2_options': q2_options, 'randomness_var': 0.1,
                'convergence_window_length': 5, 'show_progress': False}
        else:
            kwargs: dict = super()._fit_given_data_kwargs(method, data,
                                                          **user_kwargs)
        return kwargs

    def _fit_given_params_tuple(self, params: tuple, **kwargs
                                ) -> Tuple[dict, int]:
        self._check_params(params, **kwargs)
        return ({'lamb': params[0], 'chi': params[1], 'psi': params[2],
                 'loc': params[3], 'shape': params[4], 'gamma': params[5]},
                params[3].size)

    def num_scalar_params(self, d: int, copula: bool = False, **kwargs) -> int:
        vec_num: int = d if copula else 2 * d
        vec_num -= 0 if self._ASYMMETRIC else d
        return self._NUM_W_PARAMS + vec_num + self._num_shape_scalar_params(
            d=d, copula=copula)

    def fit(self, data: Union[pd.DataFrame, np.ndarray] = None,
            params: Union[Params, tuple] = None, method: str = 'mle',
            **kwargs) -> FittedContinuousMultivariate:
        """Call to fit parameters to a given dataset or to fit the distribution
        object to a set of specified parameters.

        Parameters
        ----------
        data : Union[pd.DataFrame, np.ndarray]
            Optional. The multivariate dataset to fit the distribution's
            parameters too. Not required if `params` is provided.
        params : Union[Params, tuple]
            Optional. The parameters of the distribution to fit the object
            too. These can be a Params object of the specific multivariate
            distribution or a tuple containing these parameters in the correct
            order.
        method : str
            When fitting to data only.
            The method to use when fitting the distribution to the observed
            data. Can be either 'mle' or 'em', corresponding to the Maximum
            Likelihood Estimation and Expectation-Maximization algorithms
            respectively.
            Default is 'mle'.
        kwargs:
            See below.

        Keyword Arguments
        ------------------
        bounds: tuple
            When fitting to data only.
            The bounds to use in parameter fitting / optimization, as a tuple.
        params0: Union[tuple, None]
            When fitting to data only.
            An initial estimate of the parameters to use when starting the
            optimization algorithm. These can be a Params object of the
            specific multivariate distribution or a tuple containing these
            parameters in the correct order.
            For the 'em' algorithm, if params0 specified by the user, function
            outcome ~ deterministic between runs and therefore having a
            min_retries and max_retries greater than 1 has little benefit.
        cov_method: str
            When fitting to data only.
            The method to use when estimating the sample covariance matrix.
            See CorrelationMatrix.cov for more information.
            Default value is 'pp_kendall' for 'mle' and 'laloux_pp_kendall'
            for 'em'.
        maxiter: int
            When fitting to data only.
            The maximum number of iterations to perform by the optimization
            algorithm.
            Default is 1000 for 'mle' and 100 for 'em'.
        tol: float
            When fitting to data only.
            The tolerance to use when determine convergence. For the 'em'
            algorithm, convergence is achieved when the optimization to find
            lambda, chi and psi (by maximizing Q2) successfully converges + the
            log-likelihood for the current run is our greatest observation +
            the change in log-likelihood over a given window length of runs
            is <= tol.
            For 'mle', this is the tol argument to pass to the
            differential evolution non-convex solver.
            Default value is 0.5 for 'mle' and 0.1 for 'em'.
        min_eig: Union[None, float, int]
            When fitting to data only.
            The delta / smallest positive eigenvalue to allow when enforcing
            positive definiteness via Rousseeuw and Molenberghs' technique.
            If None, the smallest eigenvalue of the sample covariance matrix is
            used.
            Default value is None.
        show_progress: bool
            When fitting to data only.
            True to display the progress of the optimization algorithm.
            Default value is False.
        kwargs:
            Any additional keyword arguments to pass to CorrelationMatrix.cov

        min_retries: int
            When fitting to data only.
            Available for the 'em' algorithm.
            The minimum number of times to re-run the EM algorithm if
            convergence is not achieved. If params0 are given, each retry will
            continue to use this as a starting point. If params0 not given, a
            new set of random values for each parameter will be generated
            before each run.
            Default value is 0.
        max_retries: int
            When fitting to data only.
            Available for the 'em' algorithm.
            The maximum number of times to re-run the EM algorithm if
            convergence is not achieved. If params0 are given, each retry will
            continue to use this as a starting point. If params0 not given, a
            new set of random values for each parameter will be generated
            before each run.
            Default value is 3.
        miniter: int
            When fitting to data only.
            Available for the 'em' algorithm.
            The minimum number of iterations to perform for each EM run,
            regardless of whether convergence has been achieved.
            Default value is 10.
        h: float
            When fitting to data only.
            Available for the 'em' algorithm.
            The h parameter to use in numerical differentiation.
            Default value is 10 ** -5.
        q2_options: dict
            When fitting to data only.
            Available for the 'em' algorithm.
            A dictionary of keyword arguments to pass to scipy's
            differential_evolution non-convex solver, when maximising Q2.
        randomness_var: float
            When fitting to data only.
            Available for the 'em' algorithm.
            A scalar value of the variance of random noise to add to
            parameters.
            Default value is 0.1.
        convergence_window_length: int
            When fitting to data only.
            Available for the 'em' algorithm.
            The window length to use when determining convergence. Convergence
            is achieved when the optimization to find lambda, chi and psi
            (by maximizing Q2) successfully converges + the log-likelihood for
            the current run is our greatest observation + the change in
            log-likelihood over a given window length of runs is <= tol.
            Default value is 5.

        Returns
        --------
        fitted_multivariate: FittedContinuousMultivariate
            A fitted distribution.
        """
        return super().fit(data=data, params=params, method=method, **kwargs)
