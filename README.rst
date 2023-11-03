=============
**SklarPy**
=============

.. image:: https://img.shields.io/pypi/v/sklarpy.svg
   :alt: PyPi Version
   :scale: 100%
   :target: https://pypi.python.org/pypi/sklarpy/

.. image:: https://github.com/sklarpy/sklarpy/actions/workflows/tests.yml/badge.svg
    :alt: Tests
    :scale: 100%
    :target: https://github.com/tfm000/sklarpy/actions/workflows/tests.yml

.. image:: https://img.shields.io/pypi/l/sklarpy.svg
   :alt: License
   :scale: 100%
   :target: https://github.com/tfm000/sklarpy/blob/main/LICENSE

.. image:: https://img.shields.io/pypi/pyversions/sklarpy.svg
   :alt: Python versions
   :scale: 100%
   :target: https://pypi.python.org/pypi/sklarpy/

.. image:: https://static.pepy.tech/personalized-badge/sklarpy?period=total&units=international_system&left_color=black&right_color=orange&left_text=Downloads
   :alt: PyPi Downloads
   :scale: 100%
   :target: https://pepy.tech/project/sklarpy

SklarPy (pronounced 'Sky-Lar-Pee' or 'Sky-La-Pie') is an open-source software for probability distribution fitting.
It contains useful tools for fitting Copula, Multivariate and Univariate probability distributions.
In addition to over 100 univariate distributions, we implement many multivariate normal mixture distributions and their copulas, including Gaussian, Student-T, Skewed-T and Generalized Hyperbolic distributions and copulas.
Named after Sklar's theorem and Abe Sklar, the American mathematician who proved that multivariate cumulative distribution functions can be expressed in terms of copulas and their marginals.

=================
**Installation**
=================

SklarPy is available on pip

::

    pip install sklarpy

Developing SklarPy
##################

To install SklarPy, along with the tool you need to develop and run tests, run the following
in your virtual environment:

::

    pip install -e .[dev]
