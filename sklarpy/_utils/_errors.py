# Contains custom errors for SklarPy

__all__ = ['SignificanceError', 'FitError', 'DiscreteError', 'SaveError', 'LoadError']


class SignificanceError(Exception):
    """Error to signify a lack of statistical significance."""
    pass


class DiscreteError(Exception):
    """Error to raise when fitting a discrete distribution to continuous data."""
    pass


class FitError(Exception):
    """Error to raise when unable to fit one or more distribution."""


class SaveError(Exception):
    """Error to raise when unable to serialize a distribution."""


class LoadError(Exception):
    """Error to raise when unable to load a serialized distribution."""
