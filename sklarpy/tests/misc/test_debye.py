# Contains code for testing SklarPy's debye function code
from sklarpy.misc import debye


def test_debye():
    """Testing debye function."""
    n_values = range(0, 10)
    x_values: list = [0, 1.5, -0.5, 4.3]

    for n in n_values:
        for x in x_values:
            val = debye(n, x)

            # checking correct datatype
            assert isinstance(val, float), f"debye({n}, {x}) is not a float."

            # testing limiting case
            if x == 0:
                assert val == 1.0, f"debye({n}, {x}) != 1.0"
