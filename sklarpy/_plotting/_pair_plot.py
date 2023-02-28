import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

__all__ = ['pair_plot']


def pair_plot(plot_df: pd.DataFrame, title: str, color: str = 'royalblue', alpha: float = 1.0,
                      figsize: tuple = (8, 8), grid: bool = True, plot_kde: bool = True, show: bool = True):

    # checking arguments
    if not isinstance(plot_df, pd.DataFrame):
        raise TypeError("plot_df must be a DataFrame")

    for str_arg in (color,):
        if not isinstance(str_arg, str):
            raise TypeError("invalid argument in pair_plot. check color is a string.")

    for numeric_arg in (alpha, ):
        if not (isinstance(numeric_arg, float) or isinstance(numeric_arg, int)):
            raise TypeError("invalid argument type in pair_plot. check alpha is a float "
                            "or integer.")
    alpha = float(alpha)

    if not (isinstance(figsize, tuple) and len(figsize) == 2):
        raise TypeError("invalid argument type in pair_plot. check figsize is a tuple of length 2.")

    for bool_arg in (grid, plot_kde, show):
        if not isinstance(bool_arg, bool):
            raise TypeError(
                "invalid argument type in pair_plot. check grid, plot_kde and show are boolean.")

    # plotting
    sns.set_style("whitegrid", {'axes.grid': grid})
    g = sns.PairGrid(plot_df, corner=(not plot_kde))
    g.fig.set_size_inches(*figsize)
    g.map_upper(sns.kdeplot, color=color)
    g.map_lower(sns.scatterplot, alpha=alpha, color=color)
    g.map_diag(sns.histplot, color=color)
    g.fig.suptitle(title)
    plt.tight_layout()

    if show:
        plt.show()