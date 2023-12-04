# Contains code for archimedean copula models
import numpy as np
import pandas as pd
from typing import Union

from sklarpy.copulas._prefit_dists import PreFitCopula
from sklarpy.copulas._fitted_dists import FittedCopula
from sklarpy.utils._params import Params
from sklarpy.copulas import MarginalFitter

__all__ = ['clayton_copula_gen', 'gumbel_copula_gen', 'frank_copula_gen']


class archimedean_copula_base_gen(PreFitCopula):
    """Base class for multivariate Archimedean copula models."""
    def fit(self, data: Union[pd.DataFrame, np.ndarray, None] = None,
            copula_params: Union[Params, tuple, None] = None,
            mdists: Union[MarginalFitter, dict, None] = None, **kwargs) \
            -> FittedCopula:
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
        method : str
            When fitting to data only.
            The method to use when fitting the distribution to the observed
            data. Can be either 'mle' or 'inverse kendall-tau.
            Data must be bivariate to use inverse kendall-tau method.
            Default is 'mle'.

       bounds: dict
            When fitting to data only.
            The bounds of the parameters you are fitting.
            Must be a dictionary with parameter names as keys and values as
            tuples of the form (lower bound, upper bound) for scalar
            parameters or values as a (d, 2) matrix for vector parameters,
            where the left hand side is the matrix contains lower bounds and
            the right hand side the upper bounds.
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
        -------
        fitted_copula: FittedCopula
            A fitted copula.
        """
        return super().fit(data=data, copula_params=copula_params,
                           mdists=mdists, **kwargs)


class clayton_copula_gen(archimedean_copula_base_gen):
    """The Multivariate Clayton copula model."""


class gumbel_copula_gen(archimedean_copula_base_gen):
    """The Multivariate Gumbel copula model."""


class frank_copula_gen(archimedean_copula_base_gen):
    """The Bivariate Frank copula model."""
