# Contains tests for Pre-Fit SklarPy copula models
import numpy as np

from sklarpy.copulas import *
from sklarpy.copulas._prefit_dists import PreFitCopula
from sklarpy.copulas._fitted_dists import FittedCopula
from sklarpy._utils import Params, FitError


def test_correct_type():
    """Testing copula distributions are all SklarPy objects."""
    print("\nTesting correct type")
    for name in distributions_map['all']:
        copula = eval(name)
        assert issubclass(type(copula), PreFitCopula), \
            f"{name} is not a child class of PreFitCopula."


def test_fit(mvt_continuous_data, mvt_discrete_data,
             pd_mvt_continuous_data, pd_mvt_discrete_data,
             mvt_mixed_data, pd_mvt_mixed_data):
    """Testing we can fit copula distributions to data."""
    print("\nTesting fit")
    for data in (mvt_continuous_data, pd_mvt_continuous_data,
                 mvt_discrete_data, pd_mvt_discrete_data, mvt_mixed_data,
                 pd_mvt_mixed_data):
        mfitter: MarginalFitter = MarginalFitter(data)
        mfitter.fit()

        for name in distributions_map['all']:
            copula = eval(name)

            for method in copula._mv_object._DATA_FIT_METHODS:
                # testing all fit methods
                if isinstance(data, np.ndarray):
                    d: int = data.shape[1]
                else:
                    d: int = len(data.columns)

                try:
                    # fitting to data
                    fcopula = copula.fit(data=data, method=method)

                    # testing fitted to correct type
                    assert issubclass(type(fcopula), FittedCopula), \
                        f"{name} is not fitted to a child class of " \
                        f"FittedCopula."

                    # testing parameters object (testing already done in
                    # multivariate tests, so not repeating all).
                    copula_params = fcopula.copula_params
                    assert issubclass(type(copula_params), Params), \
                        f"{name} fitted copula parameters are not a child " \
                        f"class of Params."
                    assert len(copula_params) > 0, \
                        f"{name} fitted copula parameter object is empty."

                    # testing we can fit distribution using parameters object.
                    params_fcopula = copula.fit(data=data,
                                                copula_params=copula_params)
                    assert issubclass(type(params_fcopula), FittedCopula), \
                        f"{name} is not a child class of FittedCopula when " \
                        f"fitted to data and copula params."

                    # testing we can fit distribution using MarginalFitter.

                    mfitter_fcopula = copula.fit(data=data, mdists=mfitter)
                    assert issubclass(type(mfitter_fcopula), FittedCopula), \
                        f"{name} is not a child class of FittedCopula when " \
                        f"fitted to data and a MarginalFitter object."

                    # testing we can fit only using copula params and
                    # MarginalFitter.
                    params_mfitter_fcopula = copula.fit(
                        copula_params=copula_params, mdists=mfitter)
                    assert issubclass(type(params_mfitter_fcopula),
                                      FittedCopula
                                      ), f"{name} is not a child class " \
                                         f"of FittedCopula when fitted to" \
                                         f" data and a MarginalFitter object."

                except FitError:
                    if method != 'inverse_kendall_tau':
                        raise
                except RuntimeError:
                    pass


