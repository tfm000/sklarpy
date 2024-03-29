# <u> Copulas </u>

This directory contains code for all copula models implemented in SklarPy.

Examples of how to use these models can be found in the `examples` folder.

## <u> SklarPy Copula Models </u>

 | Family         | Name                              | Dimensions   | SklarPy Model       |
|----------------|-----------------------------------|--------------|---------------------|
| Normal Mixture | Normal / Gaussian                 | Multivariate | gaussian_copula     | 
| Normal Mixture | Student-T                         | Multivariate | student_t_copula    |
| Normal Mixture | Skewed-T                          | Multivariate | skewed_t_copula     |
| Normal Mixture | Generalized Hyperbolic            | Multivariate | gh_copula           |
| Normal Mixture | Symmetric Generalized Hyperbolic  | Multivariate | sgh_copula          |
| Normal Mixture | Hyperbolic                        | Multivariate | hyperbolic_copula   |
| Normal Mixture | Symmetric Hyperbolic              | Multivariate | shyperbolic_copula  |
| Normal Mixture | Normal-Inverse Gaussian (NIG)     | Multivariate | nig_copula          |
| Normal Mixture | Symmetric Normal-Inverse Gaussian | Multivariate | snig_copula         |
| Normal Mixture | Marginal Hyperbolic               | Multivariate | mh_copula           |
| Normal Mixture | Symmetric Marginal Hyperbolic     | Multivariate | smh_copula          |
| Archimedean    | Clayton                           | Multivariate | clayton_copula      |
| Archimedean    | Gumbel                            | Multivariate | gumbel_copula       |
| Archimedean    | Frank                             | Bivariate    | frank_copula        |
| Numerical      | Gaussian KDE                      | Multivariate | gaussian_kde_copula |

### Implementation Status
- [x] Normal Mixture 
- [x] Numerical
- [x] Archimedean
- [ ] Vine (coming soon)
