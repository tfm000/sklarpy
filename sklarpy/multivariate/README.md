contains multivariate distributions and objects used to fit them

to implement:
- gaussian
- generalised hyperbolic and normal mixture special cases

need to be able to
- fit/provide parameter estimates, potentially via different methods. allow them to be modified to fit to copulas
- provide pdf, cdf, mc_cdf values
- generate rvs
- plots. when bivariate, allow pdf, cdf and mc_cdf plots. when multivariate n>=2, allow pairwise plots, like with copulas
- 