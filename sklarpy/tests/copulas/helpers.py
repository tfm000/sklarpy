# Contains helper functions for testing SklarPy copula code
import numpy as np

from sklarpy.copulas import *

__all__ = ['get_dist']


def get_dist(name: str, copula_params_dict: dict,
             mdists: MarginalFitter, data: np.ndarray = None
             ) -> tuple:
    """Fits a copula model to data.

    Parameters
    ----------
    name: str
        The name of the copula as in distributions_map.
    copula_params_dict: dict
        Dictionary of containing the copula parameters.
    mdists: MarginalFitter
        Fitted MarginalFitter object containing the univariate marginals.
    data: np.ndarray
        Multivariate data to fit the copula too if its params are not
        specified in copula_params_dict.

    Returns
    -------
    res: tuple
        non-fitted copula, fitted copula, copula-parameters for fitted copula.
    """
    copula = eval(name)
    if name in copula_params_dict:
        copula_params: tuple = copula_params_dict[name]
        fcopula = copula.fit(copula_params=copula_params, mdists=mdists)
    else:
        fcopula = copula.fit(data=data, mdists=mdists)
        copula_params: tuple = fcopula.copula_params.to_tuple
    return copula, fcopula, copula_params
