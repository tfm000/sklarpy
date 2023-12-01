# Contains code for producing 3D plots
from typing import Callable, Iterable
import numpy as np
import matplotlib.pyplot as plt
import warnings

from sklarpy.utils._iterator import get_iterator

__all__ = ['threeD_plot']


def threeD_plot(func: Callable, var1_range: np.ndarray, var2_range: np.ndarray,
                func_kwargs: dict = None, func_name: str = '',
                title: str = '3d Plot', color: str = 'royalblue',
                alpha: float = 1.0, figsize: tuple = (8, 8), grid: bool = True,
                axis_names: Iterable = ('variable 1', 'variable 2'),
                zlim: tuple = (None, None), show_progress: bool = True,
                show: bool = True) -> None:
    """Function able to easily implement 3D plots of functions.

    Parameters
    ----------
    func: Callable
        The callable function / method to plot.
        Must take a numpy array of size (n, 2) as values in the first argument,
         where n is an arbitrary size. Any additional arguments can be set via
         func_kwargs.
    var1_range: np.ndarray
        numpy array containing a range of values for the x1 / u1 variable to
        plot across.
    var2_range: np.ndarray
        numpy array containing a range of values for the x2 / u2 variable to
        plot across.
    func_kwargs: dict
        Optional. A dictionary containing any additional keyword / general
        arguments for the function.
        Default is None, implying no additional arguments are required.
    func_name: str
        Optional. The name of the function being evaluated. Used when labeling
        the plot figure and z axis, as well as when showing the progress of the
        plotting, if specified.
        Default is ''.
    title: str
        Optional. The matplotlib.pyplot title to use in your plot.
        Default is '3d Plot'.
    color : str
        Optional. The matplotlib.pyplot color to use in your plot.
        Default is 'royalblue'.
    alpha : float
        Optional. The matplotlib.pyplot alpha to use in your plot.
        Default is 1.0
    figsize: tuple
        Optional. The matplotlib.pyplot figsize tuple to size the overall
        figure.
        Default figsize is (8,8)
    grid : bool
        Optional. True to include a grid in the 3D plot. False for no grid.
        Default is True.
    axes_names: Iterable
        The names of your axes / variables to use in your plot.
        If provided, must be an iterable with the same length as the number of
        variables (length 2).
        Default is "('variable 1', 'variable 2')".
    zlim: tuple
        Optional. The matplotlib.pyplot bounds of the z-axis to use in your
        plot.
        Default is (None, None) -> No z-axis bounds.
    show_progress: bool
        Optional. Whether to show the progress of the plotting.
        Default is True.
    show: bool
        True to display the plot when the function is called.
        Default is True.
    """

    # checking arguments
    if not callable(func):
        raise TypeError("func must be a callable function.")

    for rng in (var1_range, var2_range):
        if not (isinstance(rng, np.ndarray) or rng.ndim == 1):
            raise TypeError("var1_range and var2_range must be 1-dimensional "
                            "numpy arrays.")

    for boolean in (grid, show_progress, show):
        if not isinstance(boolean, bool):
            raise TypeError("grid, show_progress and show must be a boolean.")

    for iterable_arg in (axis_names, zlim, figsize):
        if not (isinstance(iterable_arg, Iterable) and len(iterable_arg) == 2):
            raise TypeError("axis_names and zlim must be length 2 iterables.")

    for str_arg in (func_name, title, color):
        if not isinstance(str_arg, str):
            raise TypeError("color must be a string.")

    if (not isinstance(alpha, float) or isinstance(alpha, int)) or alpha < 0:
        raise TypeError("alpha must be a non-negative scalar value.")
    alpha = float(alpha)

    if func_kwargs is None:
        func_kwargs = {}
    elif not isinstance(func_kwargs, dict):
        raise TypeError("func_kwargs must be a dictionary.")

    # whether to show progress
    num_points: int = var1_range.size
    iterator_msg: str = f"calculating {func_name} values".replace("  ", " ")
    iterator = get_iterator(range(num_points), show_progress, iterator_msg)

    # data for plot
    Z: np.ndarray = np.full((num_points, num_points), np.nan, dtype=float)
    points: np.ndarray = np.full((num_points, 2), np.nan, dtype=float)
    for i in iterator:
        points[:, 0] = var1_range[i]
        points[:, 1] = var2_range

        try:
            Z[:, i] = func(points, **func_kwargs)
        except OverflowError:
            warnings.warn("Overflow encountered in plot")
            continue
    X, Y = np.meshgrid(var1_range, var2_range)

    # plotting
    fig = plt.figure(f"{func_name} {title} Plot", figsize=figsize)
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha)

    ax.set_xlabel(axis_names[0])
    ax.set_ylabel(axis_names[1])
    ax.set_zlabel(f"{func_name.lower()} values")
    ax.set_zlim(*zlim)
    plt.title(title)
    plt.grid(grid)

    if show:
        plt.show()
