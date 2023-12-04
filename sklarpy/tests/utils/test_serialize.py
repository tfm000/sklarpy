# Contains code for testing serialization and
# deserialization of SklarPy objects
from pathlib import Path
import pytest

from sklarpy import load
from sklarpy.univariate import normal, poisson
from sklarpy.utils._errors import LoadError


def test_load():
    """Testing SklarPy's load function."""
    # Testing if loading a non-existent file raises an error
    with pytest.raises(LoadError):
        load(r'dont\put\a\file\here\test.pickle')


def test_univariate_serialization(continuous_data, discrete_data):
    """Testing serialization and deserialization of a univariate distribution.
    """
    for dist, data in {normal: continuous_data, poisson: discrete_data
                       }.items():
        # fitting distribution
        fitted = dist.fit(data)

        # checking fitted distribution can be saved
        assert 'save' in dir(fitted), f"{fitted} can not be saved."

        save_path: str = fitted.save()

        # testing string returned when saving
        assert isinstance(save_path, str), \
            f"the path {type(fitted)} object is not returned when saved / " \
            f"serialized."

        # loading saved object
        loaded_dist = load(save_path)

        # testing loaded object is identical to the serialized one
        assert (type(loaded_dist) == type(fitted)
                and loaded_dist.params == fitted.params), \
            f"the loaded {type(fitted)} object is not idential to the " \
            f"serialized one."

        # deleting saved object
        Path(save_path).unlink()
