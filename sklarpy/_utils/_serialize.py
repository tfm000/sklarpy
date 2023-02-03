# Contains functions for loading saved SklarPy objects.
import dill
import os

from sklarpy._utils._errors import LoadError

__all__ = ['load']


def load(file: str, fix_extension: bool = True):
    """Loads pickled files.

    Parameters
    ==========
    file: str
        The file to read. Must include the full file path. The .pickle extension is optional provided fix_extension is
        True.
    fix_extension: bool
        Whether to replace any existing extension with the '.pickle' file extension. Default is True.

    See Also
    ---------
    pickle
    dill
    """
    # Input checks
    if not isinstance(file, str):
        raise TypeError("file argument must be a string.")
    if not isinstance(fix_extension, bool):
        raise TypeError("fix_extension argument must be a boolean.")

    # Changing file extension to .pickle
    file_name, extension = os.path.splitext(file)
    if fix_extension:
        extension = '.pickle'
    file = f'{file_name}{extension}'

    # Checking file exists at the specified location
    if not os.path.exists(file):
        raise LoadError(f"Unable to find file at {file}")

    # Loading file
    try:
        with open(file, 'rb') as f:
            dist = dill.load(f)
        return dist
    except Exception as e:
        raise LoadError(e)
