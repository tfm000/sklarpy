# Contains tests for the load function
import pytest
from pathlib import Path

import sklarpy
from sklarpy.univariate import normal
from sklarpy._utils import LoadError


def test_univariate_load(normal_data):
    """Test for saving and loading a univariate distribution."""
    dist = normal.fit(normal_data)
    save_location: str = dist.save()
    loaded_dist = sklarpy.load(save_location)

    # deleting saved object
    Path(save_location).unlink()


def test_univariate_load2():
    """Tests if loading a non-existent file raises an error."""
    with pytest.raises(LoadError):
        dist = sklarpy.load(r'dont\put\a\file\here\test')
