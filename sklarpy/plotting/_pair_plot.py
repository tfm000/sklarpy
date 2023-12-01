# Contains code for producing pair-plots of variables
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# from collections import deque
# import math

__all__ = ['pair_plot']


def pair_plot(plot_df: pd.DataFrame, title: str, color: str = 'royalblue',
              alpha: float = 1.0, figsize: tuple = (8, 8), grid: bool = True,
              plot_kde: bool = True, show: bool = True) -> None:
    """Produces a pair-plot of each variable in the provided dataframe.

    Parameters
    ----------
    plot_df : pd.DataFrame
        A dataframe containing the dataset to plot.
        The column names are uses to label the axes in the pair-plots.
    title : str
        The title to name your plot.
    color : str
        The matplotlib.pyplot color to use in your plots.
        Default is 'royalblue'.
    alpha : float
        The matplotlib.pyplot alpha to use in your plots.
        Default is 1.0
    figsize: tuple
        The matplotlib.pyplot figsize tuple to size the overall figure.
        Default figsize is (8,8)
    grid : bool
        True to include a grid in each pair-plot. False for no grid.
        Default is True.
    plot_kde: bool
        True to plot the KDE of your marginal distributions in the diagonal
        plots.
        Default is True.
    show: bool
        True to display the pair-plots when the method is called.
        Default is True.
    """

    # checking arguments
    if not isinstance(plot_df, pd.DataFrame):
        raise TypeError("plot_df must be a DataFrame")

    for str_arg in (color,):
        if not isinstance(str_arg, str):
            raise TypeError("invalid argument in pair_plot. check color is a "
                            "string.")

    for numeric_arg in (alpha, ):
        if not (isinstance(numeric_arg, float) or
                isinstance(numeric_arg, int)):
            raise TypeError("invalid argument type in pair_plot. check alpha is"
                            " a float or integer.")
    alpha = float(alpha)

    if not (isinstance(figsize, tuple) and len(figsize) == 2):
        raise TypeError("invalid argument type in pair_plot. check figsize is "
                        "a tuple of length 2.")

    for bool_arg in (grid, plot_kde, show):
        if not isinstance(bool_arg, bool):
            raise TypeError("invalid argument type in pair_plot. check grid, "
                            "plot_kde and show are boolean.")

    # # getting plot limits
    # lims: deque = deque()
    # n, d = plot_df.shape
    #
    # lb: int = math.floor(n * 0.01)
    # ub: int = math.ceil(n * 0.99)
    # for i in range(d):
    #     ordered_i = plot_df.iloc[:, i].sort_values()
    #     lims.append((ordered_i.iloc[lb], ordered_i.iloc[ub]))

    # plotting
    sns.set_style("whitegrid", {'axes.grid': grid})
    g = sns.PairGrid(plot_df, corner=(not plot_kde))
    g.fig.set_size_inches(*figsize)
    g.map_upper(sns.kdeplot, color=color)
    g.map_lower(sns.scatterplot, alpha=alpha, color=color)
    g.map_diag(sns.histplot, color=color)
    g.fig.suptitle(title)
    plt.tight_layout()

    # # setting limits
    # for i in range(d):
    #     g.axes[i, i].set_xlim(lims[i])
    #     g.axes[i, i].set_ylim(lims[i])

    if show:
        plt.show()
