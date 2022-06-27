from distutils.core import setup

setup(
    name='sklarpy',
    packages=['sklarpy'],
    version='0.1',
    license='MIT',
    description="Pronounced Sklar-Py, this is a Python library used for fitting and sampling from copula distributions."
                "Named after Sklar's theorem and intern Abe Sklar.",
    author='Tyler Mitchell',
    author_email='sklarpy@gmail.com',
    url='https://github.com/sklarpy/sklarpy',
    download_url='https://github.com/sklarpy/sklarpy/archive/refs/tags/v0.1.0.tar.gz',
    keywords=[
        'SklarPy',
        'Sklar',
        'Copula',
        'Copulas'
        'Probability',
        'Distribution',
        'Univariate',
        'Bivariate',
        'Multivariate',
        'Joint',
        'CDF',
        'PDF',
        'Modeling',
        'Quantitative',
        'Fitting'
        'SciPy'
    ],
    install_requires=[
        'pandas',
        'numpy',
        'scipy',
        'matplotlib'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
    ],
)
