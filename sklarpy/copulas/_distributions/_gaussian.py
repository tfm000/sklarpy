# Contains code for the gaussian copula model
import numpy as np
import pandas as pd
import scipy.stats
from typing import Union

from sklarpy.copulas._prefit_dists import PreFitCopula
from sklarpy.copulas._fitted_dists import FittedCopula
from sklarpy.utils._params import Params
from sklarpy.copulas import MarginalFitter

__all__ = ['gaussian_copula_gen']


class gaussian_copula_gen(PreFitCopula):
    """The Multivariate Gaussian copula model."""
    def _g_to_u(self, g: np.ndarray, copula_params_tuple: Union[Params, tuple]
                ) -> np.ndarray:
        return scipy.stats.norm.cdf(x=g)

    def _u_to_g(self, u: np.ndarray, copula_params: Union[Params, tuple]) \
            -> np.ndarray:
        return scipy.stats.norm.ppf(q=u)

    def _h_logpdf_sum(self, g: np.ndarray, copula_params: Union[Params, tuple]
                      ) -> np.ndarray:
        return scipy.stats.norm.logpdf(g).sum(axis=1)

    def fit(self, data: Union[pd.DataFrame, np.ndarray, None] = None,
            copula_params: Union[Params, tuple, None] = None,
            mdists: Union[MarginalFitter, dict, None] = None, **kwargs
            ) -> FittedCopula:
        """Fits the overall joint distribution to a given dataset or user
        provided parameters.

        For the Normal / Gaussian copula, we use the closed-form MLE solution
        for all parameter fitting / optimization.

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
        corr_method: str
            When fitting to data only.
            The method to use when estimating the sample correlation matrix.
            See CorrelationMatrix.corr for more information.
            Default value is 'laloux_pp_kendall'.
        kwargs:
            kwargs for CorrelationMatrix.corr

        Returns
        -------
        fitted_copula: FittedCopula
            A fitted copula.
        """
        return super().fit(data=data, copula_params=copula_params,
                           mdists=mdists, **kwargs)
