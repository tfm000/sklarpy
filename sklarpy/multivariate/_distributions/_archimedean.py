# Contains code for Archimedean copula models
# Unlike other multivariate distributions, these are not intended to be used
# outside their respective copula distributions
import numpy as np
import pandas as pd
from abc import abstractmethod
from typing import Union, Tuple, Callable
import scipy.stats
import scipy.optimize

from sklarpy.multivariate._prefit_dists import PreFitContinuousMultivariate
from sklarpy.multivariate._fitted_dists import FittedContinuousMultivariate
from sklarpy.utils._params import Params
from sklarpy.utils._iterator import get_iterator
from sklarpy.utils._errors import FitError
from sklarpy.misc import debye

__all__ = ['multivariate_clayton_gen', 'multivariate_gumbel_gen',
           'bivariate_frank_gen']


class multivariate_archimedean_base_gen(PreFitContinuousMultivariate):
    """Base class for multivariate Archimedean models."""
    _DATA_FIT_METHODS = ("mle", 'inverse_kendall_tau')
    _DEFAULT_STRICT_BOUNDS: tuple
    _DEFAULT_BOUNDS: tuple
    _N_PARAMS: int

    def _get_dim(self, params: tuple) -> int:
        return params[-1]

    @abstractmethod
    def _param_range(self, d: int) -> Tuple[Tuple[float, float], np.ndarray]:
        """Returns the parameter range for which the Archimedean copula is
        defined.

        Parameters
        ----------
        d: int
            The dimension / number of variables.

        Returns
        -------
        bounds, excluded : tuple
            The inclusive bounds of any parameters, An array of values within
            the bounds which the dist is not defined for.
        """

    def _check_params(self, params: tuple, **kwargs) -> None:
        # checking correct number of params passed
        super()._check_params(params)

        # checking valid theta and dimensions parameters
        d = params[-1]
        if not isinstance(d, int):
            raise TypeError("d (dimensions parameter) must be an integer.")
        elif d < 2:
            raise ValueError("d (dimensions parameter) must be greater than "
                             "or equal to 2.")

        theta = params[0]
        bounds, excluded = self._param_range(d=d)
        if not (isinstance(theta, float) or isinstance(theta, int)):
            raise TypeError("theta must be a scalar value.")
        elif (not bounds[0] <= theta <= bounds[1]) or theta in excluded:
            excluded_msg: str = " " if len(excluded) == 0 \
                else f" cannot be any of {excluded} and "
            raise ValueError(f"theta parameter{excluded_msg}must lie within "
                             f"{bounds[0]} <= theta <= {bounds[1]} when d={d}."
                             f" However, theta={theta}")

    @abstractmethod
    def _generator(self, u: np.ndarray, params: tuple) -> np.ndarray:
        """The generator function associated with the Archimedean copula.

        We take the generator functions specified by McNeil, Frey and
        Embrechts.

        Parameters
        ----------
        u: Union[pd.DataFrame, np.ndarray]
            The cdf / pseudo-observation values to evaluate the pdf function of
            the copula distribution at. Each ui must be in the range (0, 1)
            and should be the cdf values of the univariate marginal
            distribution of the random variable xi.
        params: Union[pd.DataFrame, tuple]
            The parameters which define the multivariate model. These can be a
            Params object of the specific multivariate distribution or a tuple
            containing these parameters in the correct order.

        Returns
        -------
        generator_values: np.ndarray
            values of the generator function.
        """

    @abstractmethod
    def _generator_inverse(self, t: np.ndarray, params: tuple) -> np.ndarray:
        """The inverse of the generator function associated with the
        Archimedean copula.

        We take the generator functions specified by McNeil, Frey and
        Embrechts.

        Parameters
        ----------
        t: np.ndarray
            Values of the generator function.
        params: Union[pd.DataFrame, tuple]
            The parameters which define the multivariate model. These can be a
            Params object of the specific multivariate distribution or a tuple
            containing these parameters in the correct order.

        Returns
        -------
        inverse_generator_values: np.ndarray
            values of the inverse generator function.
        """

    def _cdf(self, x: np.ndarray, params: tuple, **kwargs) -> np.ndarray:
        shape: tuple = x.shape

        show_progress: bool = kwargs.get('show_progress', True)
        iterator = get_iterator(range(shape[1]), show_progress,
                                "calculating cdf values")

        t: np.ndarray = np.zeros((shape[0], ), dtype=float)
        for i in iterator:
            t += self._generator(u=x[:, i], params=params)
        return self._generator_inverse(t=t, params=params)

    def _G_hat(self, t: np.ndarray, params: tuple) -> np.ndarray:
        """Function used when generating random variates from the Archimedean
        distribution.

        See Also
        --------
        McNeil Frey and Embrechts:
            Algorithm 5.48

        Returns
        -------
        G_hat_values: np.ndarray
            values of the G_hat function.
        """
        return self._generator_inverse(t=t, params=params)

    @abstractmethod
    def _v_rvs(self, size: int, params: tuple) -> np.ndarray:
        """Returns random variates, generated from univariate distribution G.

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
            Algorithm 5.48

        Returns
        -------
        v_rvs: : np.ndarray
            univariate array of random variables, sampled from distribution G.
        """

    def _rvs(self, size: int, params: tuple) -> np.ndarray:
        v: np.ndarray = self._v_rvs(size=size, params=params)
        d: int = params[-1]
        x: np.ndarray = np.random.uniform(size=(size, d))
        t: np.ndarray = -np.log(x) / v
        return self._G_hat(t=t, params=params)

    def _get_bounds(self, data: np.ndarray, as_tuple: bool, **kwargs) \
            -> Union[dict, tuple]:
        d: int = data.shape[1]
        theta_bounds: tuple = self._DEFAULT_STRICT_BOUNDS if d > 2 \
            else self._DEFAULT_BOUNDS
        default_bounds: dict = {'theta': theta_bounds}
        return super()._get_bounds(default_bounds, d, as_tuple, **kwargs)

    def _get_params0(self, data: np.ndarray, bounds: tuple, copula: bool,
                     **kwargs) -> tuple:
        theta0: float = np.random.uniform(*bounds[0])
        return theta0, data.shape[1]

    def _theta_to_params(self, theta: np.ndarray, d: int, **kwargs) -> tuple:
        return theta[0], d

    def _params_to_theta(self, params: tuple, **kwargs) -> np.ndarray:
        return np.array([params[0]], dtype=float)

    def _get_mle_objective_func_kwargs(self, data: np.ndarray, **kwargs,
                                     ) -> dict:
        _, excluded = self._param_range(d=data.shape[1])
        con_shape: tuple = excluded.shape
        if excluded.size == 0:
            # no equality constraints
            constraints: tuple = tuple()
        else:
            # adding equality constraints
            con: Callable = lambda theta: np.abs(np.full(con_shape, theta[0])
                                                 - excluded)
            nlc: scipy.optimize.NonlinearConstraint = \
                scipy.optimize.NonlinearConstraint(
                    con, np.full(con_shape, 10**-5),
                    np.full(con_shape, np.inf))
            constraints: tuple = (nlc, )
        return {'d': data.shape[1], 'constraints': constraints}

    def _inverse_kendall_tau_calc(self, kendall_tau: float) -> float:
        """Estimates the theta parameter analytically, by using the
        Kendall-Tau relationship specified by McNeil Frey and Embrechts. Only
        applicable for bivariate Clayton, Gumbel and Frank copulas.

        Parameters
        ----------
        kendall_tau : float
            The Kendall-Tau correlation value.

        See Also
        --------
        McNeil Frey and Embrechts:
            Proposition 5.45

        Returns
        -------
        theta_hat: : float
            Estimate of theta parameter
        """

    def _inverse_kendall_tau(self, data: np.ndarray, **kwargs) \
            -> Tuple[tuple, bool]:
        """Estimates the theta parameter analytically, by using the
        Kendall-Tau relationship specified by McNeil Frey and Embrechts. Only
        applicable for bivariate Clayton, Gumbel and Frank copulas.

        Parameters
        ----------
        data: np.ndarray
            An array of multivariate data to optimize parameters over using
            the inverse Kendall-Tau algorithm.

        See Also
        --------
        McNeil Frey and Embrechts:
            Proposition 5.45

        Returns
        -------
        res: Tuple[tuple, bool]
            The parameters optimized to fit the data,
            True if theta is not inf false otherwise.
        """
        d: int = data.shape[1]
        if d != 2:
            raise FitError("Archimedean copulas can only be fit using inverse"
                           " kendall's tau when the number of variables is"
                           " exactly 2.")

        kendall_tau: float = scipy.stats.kendalltau(data[:, 0],
                                                    data[:, 1]).statistic
        theta: float = self._inverse_kendall_tau_calc(kendall_tau=kendall_tau)
        params: tuple = (theta, d)
        try:
            self._check_params(params=params)
        except Exception:
            raise FitError(f"cannot fit {self.name} using inverse kendall's "
                           f"tau method. This is because the fitted theta "
                           f"parameter likely lies outside its permitted "
                           f"range.")
        return (theta, d), ~np.isinf(theta)

    def _fit_given_data_kwargs(self, method: str, data: np.ndarray,
                               **user_kwargs) -> dict:
        if method == 'inverse_kendall_tau':
            return {'copula': True}
        return super()._fit_given_data_kwargs(method=method, data=data,
                                              **user_kwargs)

    def _fit_given_params_tuple(self, params: tuple, **kwargs) \
            -> Tuple[dict, int]:
        d: int = params[-1]
        return {'theta': params[0], 'd': d}, d

    def num_scalar_params(self, d: int, copula: bool, **kwargs) -> int:
        return self._N_PARAMS

    def fit(self, data: Union[pd.DataFrame, np.ndarray] = None,
            params: Union[Params, tuple] = None, method: str = 'mle',
            **kwargs) -> FittedContinuousMultivariate:
        """Call to fit parameters to a given dataset or to fit the
        distribution object to a set of specified parameters.

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
            data. Can be either 'mle' or 'inverse kendall-tau.
            Data must be bivariate to use inverse kendall-tau method.
            Default is 'mle'.
        kwargs:
            See below.

        Keyword arguments
        ------------------
        bounds: dict
            When fitting to data only.
            The bounds of the parameters you are fitting.
            Must be a dictionary with parameter names as keys and values as
            tuples of the form (lower bound, upper bound) for scalar
            parameters or values as a (d, 2) matrix for vector parameters,
            where the left hand side is the matrix contains lower bounds and
            the right hand side the upper bounds.
        show_progress: bool
            When fitting to data only.
            Available for 'mle' algorithm.
            Prints the progress of this algorithm.
            Default is False.
        maxiter: int
            When fitting to data only.
            Available for 'mle' algorithm.
            The maximum number of iterations to perform by the differential
            evolution solver.
            Default value is 1000.
        tol: float
            When fitting to data only.
            Available for 'mle' algorithm.
            The tolerance to use when determining convergence.
            Default value is 0.5.
        params0: Union[Params, tuple]
            When fitting to data only.
            Available for 'mle' algorithm.
            An initial estimate of the parameters to use when starting the
            optimization algorithm. These can be a Params object of the
            specific multivariate distribution or a tuple containing these
            parameters in the correct order.

        Returns
        --------
        fitted_multivariate: FittedContinuousMultivariate
            A fitted Archimedean distribution.
        """
        return super().fit(data=data, params=params, method=method, **kwargs)


class multivariate_clayton_gen(multivariate_archimedean_base_gen):
    """Multivariate distribution for the multivariate Clayton copula model."""
    _DEFAULT_STRICT_BOUNDS = (0, 100.0)
    _DEFAULT_BOUNDS = (-1, 100.0)
    _N_PARAMS = 2

    def _param_range(self, d: int) -> Tuple[Tuple[float, float], np.ndarray]:
        lb: float = 0.0 if d > 2 else -1
        return (lb, np.inf), np.array([0.0])

    def _generator(self, u: np.ndarray, params: tuple) -> np.ndarray:
        theta: float = params[0]
        return (np.power(u, -theta) - 1) / theta

    def _generator_inverse(self, t: np.ndarray, params: tuple) -> np.ndarray:
        theta: float = params[0]
        return np.power((theta * t) + 1, -1 / theta)

    def _G_hat(self, t: np.ndarray, params: tuple) -> np.ndarray:
        theta: float = params[0]
        return np.power(t + 1, -1 / theta)

    def _v_rvs(self, size: int, params: tuple) -> np.ndarray:
        theta = params[0]
        if theta < 0:
            return np.full((size, 1), np.nan)
        return scipy.stats.gamma.rvs(a=1/theta, scale=1, size=(size, 1))

    def _inverse_kendall_tau_calc(self, kendall_tau: float) -> float:
        return 2 * kendall_tau / (1 - kendall_tau)

    def _logpdf(self, x: np.ndarray, params: tuple,  **kwargs) -> np.ndarray:
        theta, d = params

        # common calculations
        theta_inv: float = 1/theta

        # calculating evaluating generator function
        gen_sum: np.ndarray = np.zeros((x.shape[0], ), dtype=float)
        log_cs: float = 0.0
        log_gs: float = 0.0
        for i in range(d):
            gen_val: np.ndarray = self._generator(u=x[:, i], params=params)
            gen_sum += gen_val

            log_cs += np.log(theta_inv + d - i)
            log_gs += np.log((theta*gen_val) + 1)

        return (d * np.log(theta)) \
               - ((theta_inv + d) * np.log((theta * gen_sum) + 1)) + log_cs \
               + ((theta_inv + 1) * log_gs)


class multivariate_gumbel_gen(multivariate_archimedean_base_gen):
    """Multivariate distribution for the multivariate Gumbel copula model."""
    _DEFAULT_STRICT_BOUNDS = (1.001, 100.0)
    _DEFAULT_BOUNDS = _DEFAULT_STRICT_BOUNDS
    _N_PARAMS = 2

    def _param_range(self, d: int) -> Tuple[Tuple[float, float], np.ndarray]:
        return (self._DEFAULT_BOUNDS[0], np.inf), np.array([])

    def _generator(self, u: np.ndarray, params: tuple) -> np.ndarray:
        theta: float = params[0]
        return np.power(-np.log(u), theta)

    def _generator_inverse(self, t: np.ndarray, params: tuple) -> np.ndarray:
        theta: float = params[0]
        return np.exp(-np.power(t, 1/theta))

    def _v_rvs(self, size: int, params: tuple) -> np.ndarray:
        # For the special case of the Gumbel copula,
        # we have V ~ St(1 / theta, 1, c, 0)
        theta: float = params[0]
        alpha: float = 1 / theta
        beta: float = 1.0
        c: float = np.power(np.cos(np.pi/(2*theta)), theta)
        mu: float = 0.0

        # simulating X ~ St(alpha, beta, 1, 0) rvs
        u: np.ndarray = np.random.uniform(-0.5*np.pi, 0.5*np.pi, size)
        w: np.ndarray = scipy.stats.expon.rvs(size=size)
        zeta: float = -beta * np.tan(np.pi * alpha * 0.5)
        if alpha != 1.0:
            xi: float = np.arctan(-zeta) / alpha
            x: np.ndarray = (((1 + (zeta ** 2)) ** (0.5 / alpha))
                             * np.sin(alpha * (u + xi))
                             * np.power(
                        np.cos(u - (alpha * (u + xi)))/w, (1 - alpha) / alpha)
                             ) / np.power(np.cos(u), 1 / alpha)

        else:
            xi: float = np.pi * 0.5
            x: np.ndarray = ((((np.pi / 2) + (beta * u)) * np.tan(u))
                             - (beta
                                * np.log(
                                ((np.pi / 2) * w * np.cos(u))
                                / ((np.pi /2) + (beta * u)))
                                )) / xi
            mu += (2 / np.pi) * beta * c * np.log(c)

        # simulating Y ~S t(alpha, beta, c, mu) rvs
        return ((c * x) + mu).reshape((size, 1))

    def _inverse_kendall_tau_calc(self, kendall_tau: float) -> float:
        return 1 / (1 - kendall_tau)

    def _DK_psi(self, t: np.ndarray, params: tuple, K: int) -> np.ndarray:
        if K == 0:
            return self._generator_inverse(t=t, params=params)
        theta: float = params[0]
        theta_inv: float = 1/theta

        DK_psi: np.ndarray = np.zeros(t.shape, dtype=float)
        for j in range(K):
            prod = np.array([theta_inv - i + 1
                             for i in range(1, K - j + 1)]).prod()
            DK_psi -= self._DK_psi(t=t, params=params, K=j) \
                      * np.power(t, theta_inv - K + j) * prod
        return DK_psi

    def _logpdf(self, x: np.ndarray, params: tuple,  **kwargs) -> np.ndarray:
        theta, d = params

        # calculating evaluating generator function
        gen_sum: np.ndarray = np.zeros((x.shape[0],), dtype=float)
        log_gs: float = 0.0
        for i in range(d):
            gen_val: np.ndarray = self._generator(u=x[:, i], params=params)
            gen_sum += gen_val

            log_gs -= (((1/theta) - 1) * np.log(gen_val)) + np.log(x[:, i])

        # calculating d-th derivative of generator inverse
        Dd_psi: np.ndarray = self._DK_psi(t=gen_sum, params=params, K=d)
        return np.log(np.abs(Dd_psi)) + (d*np.log(theta)) + log_gs


class bivariate_frank_gen(multivariate_archimedean_base_gen):
    """Bivariate distribution for the bivariate Gumbel copula model."""
    _DEFAULT_STRICT_BOUNDS = (-100.0, 100.0)
    _DEFAULT_BOUNDS = _DEFAULT_STRICT_BOUNDS
    _N_PARAMS = 2

    def _param_range(self, d: int) -> Tuple[Tuple[float, float], np.ndarray]:
        return (-np.inf, np.inf), np.array([0.0])

    def _generator(self, u: np.ndarray, params: tuple) -> np.ndarray:
        theta: float = params[0]
        return -np.log((np.exp(-theta*u) - 1) / (np.exp(-theta) - 1))

    def _generator_inverse(self, t: np.ndarray, params: tuple) -> np.ndarray:
        theta: float = params[0]
        return -(theta**-1) * np.log(1 + np.exp(-t) * (np.exp(-theta) - 1))

    def _rvs(self, size: int, params: tuple) -> np.ndarray:
        theta: float = params[0]
        rvs: np.ndarray = np.random.uniform(size=(size, 2))
        rvs[:, 1] = - (theta**-1) * np.log(
            1 + (((1 - np.exp(-theta)) * rvs[:, 1])
                 / (
                         (rvs[:, 1] * (np.exp(-theta*rvs[:, 0]) - 1))
                         - np.exp(-theta*rvs[:, 0]))
                 )
        )
        return rvs

    def __root_to_solve(self, theta: float, kendall_tau: float) -> float:
        """The function root to numerically solve when calculating the theta
        parameter in the inverse Kendall-Tau algorithm.

        Parameters
        ----------
        theta : float
            A value of the theta parameter defining the Frank copula.
        kendall_tau : float
            The Kendall-Tau rank correlation value derived from the data.

        Returns
        -------
        func_value : float
            The value of the function.
        """
        return 1 - (4 * (theta ** -1) * (1 - debye(1, theta))) - kendall_tau

    def _inverse_kendall_tau_calc(self, kendall_tau: float) -> float:
        factor = 2.
        left, right = -factor, factor

        while self.__root_to_solve(left, kendall_tau) > 0.:
            left, right = left * factor, left
        # left is now such that func(left) <= tau
        # if right has changed, then func(right) > tau

        while self.__root_to_solve(right, kendall_tau) < 0.:
            left, right = right, right * factor
        # right is now such that func(right) >= tau

        return scipy.optimize.brentq(self.__root_to_solve, left, right,
                                     args=(kendall_tau,))

    def _pdf(self, x: np.ndarray, params: tuple, **kwargs) -> np.ndarray:
        theta: float = params[0]

        numerator: np.ndarray = theta * (np.exp(theta) - 1) \
                                * np.exp(theta * (x.sum(axis=1) + 1))
        denominator: np.ndarray = np.power(
            np.exp(theta * x.sum(axis=1))
            - np.exp(theta * (x[:, 0] + 1))
            - np.exp(theta * (x[:, 1] + 1))
            + np.exp(theta), 2)
        return numerator / denominator
