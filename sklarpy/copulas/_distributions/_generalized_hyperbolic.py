# Contains code for the generalized hyperbolic copula model
import numpy as np
import pandas as pd
from typing import Union, Callable

from sklarpy.copulas._prefit_dists import PreFitCopula
from sklarpy.copulas._fitted_dists import FittedCopula
from sklarpy.utils._params import Params
from sklarpy.copulas import MarginalFitter
from sklarpy.univariate import gh

__all__ = ['gen_hyperbolic_copula_gen']


class gen_hyperbolic_copula_gen(PreFitCopula):
    """The Multivariate Generalized Hyperbolic copula model."""
    def _u_g_pdf(self, func: Callable, arr: np.ndarray,
                 copula_params: Union[Params, tuple], **kwargs) -> np.ndarray:
        shape: tuple = arr.shape
        output: np.ndarray = np.full(shape, np.nan)
        for i in range(shape[1]):
            gh_params: tuple = (
                copula_params[0], copula_params[1], copula_params[2],
                0.0, 1.0, float(copula_params[-1][i])
            )
            output[:, i] = func(arr[:, i], gh_params, **kwargs)
        return output

    def _g_to_u(self, g: np.ndarray, copula_params: Union[Params, tuple]) \
            -> np.ndarray:
        return self._u_g_pdf(func=gh.cdf_approx, arr=g,
                             copula_params=copula_params, num_points=10)

    def _u_to_g(self, u: np.ndarray, copula_params: Union[Params, tuple]):
        return self._u_g_pdf(func=gh.ppf_approx, arr=u,
                             copula_params=copula_params, num_points=10)

    def _h_logpdf_sum(self, g: np.ndarray, copula_params: Union[Params, tuple]
                      ) -> np.ndarray:
        pdf_vals: np.ndarray = self._u_g_pdf(func=gh.pdf, arr=g,
                                             copula_params=copula_params)
        return np.log(pdf_vals).sum(axis=1)

    def fit(self, data: Union[pd.DataFrame, np.ndarray, None] = None,
            copula_params: Union[Params, tuple, None] = None,
            mdists: Union[MarginalFitter, dict, None] = None, **kwargs
            ) -> FittedCopula:
        """Fits the overall joint distribution to a given dataset or user
        provided parameters.

        Parameters
        ----------
        data : Union[pd.DataFrame, np.ndarray, None]
            The multivariate dataset to fit the distribution's parameters too.
            Not required if `copula_params` and `mdists` provided.
        copula_params: Union[Params, tuple, None]
            The parameters of the multivariate distribution used to specify
            your copula distribution. Can be a Params object of the
            specific multivariate distribution or a tuple containing these
            parameters in the correct order. If not passed, user must provide
            a dataset to fit too.
        mdists : Union[MarginalFitter, dict, None]
            The fitted marginal distributions of each random variable.
            Must be a fitted MarginalFitter object or a dictionary with
            numbered indices as values and fitted SklarPy univariate
            distributions as values. The dictionary indices must correspond
            to the indices of the variables. If not passed, user must provide
            a dataset to fit too.
        kwargs:
            kwargs to pass to MarginalFitter.fit and / or the relevant
            multivariate distribution's .fit method
            See below

        Keyword Arguments
        -----------------
        univariate_fitter_options: dict
            User provided arguments to use when fitting each marginal
            distribution. See MarginalFitter.fit documentation for more.
        show_progress: bool
            True to show the progress of your fitting.
        method: str
            The method to use when fitting the copula distribution to data.
            Can be either 'mle' or 'em'. Default is 'mle'.

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
        corr_method: str
            When fitting to data only.
            The method to use when estimating the sample correlation matrix.
            See CorrelationMatrix.corr for more information.
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
            If None, the smallest eigenvalue of the sample correlation matrix
            is used.
            Default value is None.
        show_progress: bool
            When fitting to data only.
            True to display the progress of the optimization algorithm.
            Default value is False.
        kwargs:
            Any additional keyword arguments to pass to CorrelationMatrix.corr

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
        -------
        fitted_copula: FittedCopula
            A fitted copula.
        """
        return super().fit(data=data, copula_params=copula_params,
                        mdists=mdists, **kwargs)
