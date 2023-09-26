# Contains code for testing SklarPy's modified bessel function code
import numpy as np
from typing import Callable
import pytest

from sklarpy.misc import kv


def test_kv():
    """Testing the Modified Bessel function of the 2nd kind."""
    v_values = [0, 0.1, 4.3, 9.7]
    z_values = [0, *np.random.uniform(0, 200, 100)]

    for func_str in ('kv', 'logkv'):
        # checking function is implemented
        assert func_str in dir(kv), \
            f"{func_str} is not implemented for Modified Bessel function of " \
            f"the 2nd kind."

        func: Callable = eval(f'kv.{func_str}')

        # evaluating for different values of v and z
        for z in z_values:
            for v in v_values:
                func_val = func(v, z)

                # checking correct type
                assert isinstance(func_val, float), \
                    f'{func_str}({v}, {z}) is not a float.'

                # checking same value as -v
                assert func_val == func(-v, z), \
                    f'{func_str}({v}, {z}) != {func_str}(-{v}, z).'

                # checking non-nan values when z >= 0
                assert not np.isnan(func_val), f'{func_str}({v}, {z}) is nan.'

                # checking nan values when z < 0
                if z != 0:
                    assert np.isnan(func(v, -z)), \
                        f'{func_str}({v}, -{z}) is not nan.'

        # checking fails for non-scalar values
        with pytest.raises(TypeError, match="all arguments / keyword "
                                            "arguments must be scalars."):
            func(0.1, z_values)

        with pytest.raises(TypeError, match="all arguments / keyword "
                                            "arguments must be scalars."):
            func(v_values, 5)
