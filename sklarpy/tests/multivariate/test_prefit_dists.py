# Contains tests for Pre-Fit SklarPy multivariate distributions
import numpy as np
import pytest

from sklarpy.multivariate import *
from sklarpy.multivariate._prefit_dists import PreFitContinuousMultivariate
from sklarpy.multivariate._fitted_dists import FittedContinuousMultivariate
from sklarpy._utils import Params


def test_correct_type():
    """Testing distributions are all SklarPy objects."""
    for name in distributions_map['all']:
        dist = eval(name)
        assert issubclass(type(dist), PreFitContinuousMultivariate), \
            f"{name} is not a child class of PreFitContinuousMultivariate"


def test_fit_to_data(mvt_continuous_data, mvt_discrete_data,
                     pd_mvt_continuous_data, pd_mvt_discrete_data,
                     mv_dists_to_test):
    """Testing we can fit distributions to data."""
    for name in mv_dists_to_test:
        dist = eval(name)

        # fitting to both continuous and discrete data,
        # in both numpy and pandas format
        for data in (mvt_continuous_data, pd_mvt_continuous_data,
                     mvt_discrete_data, pd_mvt_discrete_data):
            if isinstance(data, np.ndarray):
                d: int = data.shape[1]
            else:
                d: int = len(data.columns)

            try:
                # fitting to data
                fitted = dist.fit(data=data)
            except RuntimeError:
                continue

            # testing parameters object
            params = fitted.params
            assert issubclass(type(params), Params), \
                f"{name} fitted parameters are not a child class of Params."
            assert params.name == name, \
                f"{name} fitted parameters is not the correct type."
            assert len(params) > 0, f"{name} fitted parameter object is empty."

            vector_attributes: tuple = ('loc', 'mean', 'gamma')
            matrix_attributes: tuple = ('cov', 'corr')
            scale_attributes: tuple = ('dof', 'chi', 'psi', 'lamb', 'theta')
            to_obj_attributes: tuple = ('dict', 'tuple', 'list')

            for vect_str in vector_attributes:
                if vect_str in dir(params):
                    vect = eval(f'params.{vect_str}')
                    assert isinstance(vect, np.ndarray), \
                        f"{vect_str} fitted parameter is not an array for " \
                        f"{name}."
                    assert vect.size == d, \
                        f"{vect_str} fitted parameter does not contain the " \
                        f"correct number of elements for {name}."
                    assert vect.shape == (d, 1), \
                        f"{vect_str} fitted parameter is not of {(d, 1)} " \
                        f"shape for {name}."
                    assert np.isnan(vect).sum() == 0, \
                        f"{vect_str} fitted parameter contains nan values " \
                        f"for {name}."

            for mat_str in matrix_attributes:
                if mat_str in dir(params):
                    mat = eval(f'params.{mat_str}')
                    assert isinstance(mat, np.ndarray), \
                        f"{mat_str} fitted parameter is not an array " \
                        f"for {name}."
                    assert mat.shape == (d, d), \
                        f"{mat_str} fitted parameter is not of {(d, d)} " \
                        f"shape for {name}."
                    assert np.isnan(mat).sum() == 0, \
                        f"{mat_str} fitted parameter contains nan values " \
                        f"for {name}."

            for scale_str in scale_attributes:
                if scale_str in dir(params):
                    scale = eval(f'params.{scale_str}')
                    assert (isinstance(scale, float)
                            or isinstance(scale, int)), \
                        f"{scale_str} fitted parameter is not a scalar " \
                        f"value for {name}."
                    assert not np.isnan(scale),\
                        f"{scale_str} fitted parameter is nan for {name}."

            for obj_str in to_obj_attributes:
                assert f'to_{obj_str}' in dir(params), \
                    f"to_{obj_str} attribute does not exist for {name}"
                obj_target_type = eval(obj_str)
                obj = eval(f'params.to_{obj_str}')
                assert isinstance(obj, obj_target_type), \
                    f"to_{obj_str} attribute does not return a " \
                    f"{obj_target_type} for {name}."
                assert len(obj) == len(params), \
                    f"to_{obj_str} attribute does not contain the correct " \
                    f"number of parameters for {name}."

            # testing we can fit distribution using parameters object.
            params_fitted = dist.fit(params=params)

            # testing we can fit distribution using tuple object.
            tuple_fitted = dist.fit(params=params.to_tuple)

            # testing for errors if incorrect params object provided
            with pytest.raises(
                    TypeError,
                    match=f"if params provided, must be a "
                          f"{dist._params_obj} type or tuple of length "
                          f"{dist.num_params}"):
                dist.fit(params=range(1000))
