# Contains code for the gaussian kde copula model
import numpy as np
import pandas as pd
from typing import Union

from sklarpy.copulas._prefit_dists import PreFitCopula
from sklarpy.copulas._fitted_dists import FittedCopula
from sklarpy.copulas import MarginalFitter
from sklarpy.utils._params import Params

__all__ = ['gaussian_kde_copula_gen']


class gaussian_kde_copula_gen(PreFitCopula):
    """The Multivariate Gaussian KDE copula model."""

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

        bw_method : Union[str, int, float, Callable, None]
            The method used to calculate the estimator bandwidth.
            See scipy.stats.gaussian_kde for options.
        weights: Union[np.ndarray, None]
            weights of datapoints.
            See scipy.stats.gaussian_kde for options.

        Returns
        -------
        fitted_copula: FittedCopula
            A fitted copula.
        """
        return super().fit(data=data, copula_params=copula_params,
                           mdists=mdists, **kwargs)
