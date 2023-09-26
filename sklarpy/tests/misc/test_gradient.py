# Contains code for testing SklarPy's numerical derivative functions
import numpy as np
import pandas as pd

from sklarpy.misc import gradient_1d
from sklarpy.tests.misc.helpers import XCubed, Exp, Log


def test_gradient_1d():
    """Testing gradient_1d function."""
    funcs: list = [XCubed, Exp, Log]
    x_values = [0, 1.5, -0.5, 4.3, 9.7, 11, -8]
    datatypes = [np.asarray, list, pd.DataFrame, pd.Series, set]

    for func in funcs:
        # log(x) undefined for non-positive x
        x_vals = [x for x in x_values if not (func == Log and x <= 0)]

        # calculating true values
        dfdx: np.ndarray = func.dfdx(x_vals)

        # checking different datatypes
        for datatype in datatypes:
            x_vals_dtype = datatype(x_vals)
            dfdx_approx = gradient_1d(func.f, x_vals_dtype, 10**-5)

            assert isinstance(dfdx_approx, np.ndarray), \
                f"gradient_1d does not return an array for {func} when x is " \
                f"{datatype}"

            assert len(dfdx_approx) == len(x_vals), \
                f"gradient_1d output is not the same length as its input " \
                f"for {func} when x is {datatype}"

            assert np.isnan(dfdx_approx).sum() == 0, \
                f"nan values returned by gradient_1d for {func} when x " \
                f"is {datatype}"

            # comparing against true values
            if datatype == set:
                # set changes ordering of values
                dfdx = func.dfdx(list(x_vals_dtype))

            assert np.allclose(dfdx, dfdx_approx), \
                f"gradient_1d values are poor approximations of {func} " \
                f"when x is {datatype}"
