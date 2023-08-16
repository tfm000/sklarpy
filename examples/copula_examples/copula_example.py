import numpy as np
import pandas as pd

my_loc = np.array([1, -3], dtype=float)
my_shape = np.array([[1, 0.7], [0.7, 1]], dtype=float)
my_chi = 1.7
my_psi = 4.5
my_gamma = np.array([2.3, -4.3], dtype=float)
my_params = (my_chi, my_psi, my_loc, my_shape, my_gamma)

from sklarpy.multivariate import multivariate_hyperbolic

num_generate: int = 1000
rvs: np.ndarray = multivariate_hyperbolic.rvs(num_generate, my_params)
rvs_df: pd.DataFrame = pd.DataFrame(rvs, columns=['Wife Age', 'Husband Age'], dtype=float)
# multivariate_hyperbolic.pdf_plot(my_params)
print(rvs_df)

from sklarpy.copulas import gaussian_copula as copula
fitted_copula = copula.fit(data=rvs_df, univariate_fitter_options={'significant': False, 'distributions': ['student_t']})
fitted_copula.pdf_plot()
fitted_copula.copula_pdf_plot()
# fitted_copula.pdf_plot(show=False)
# multivariate_hyperbolic.pdf_plot(my_params, show=False)
import matplotlib.pyplot as plt
plt.show()
breakpoint()