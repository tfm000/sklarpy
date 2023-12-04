# Contains code for loading and saving SklarPy objects.
import dill
import os
from pathlib import Path

from sklarpy.utils._errors import LoadError, SaveError

__all__ = ['load', 'Savable']


def load(file: str, fix_extension: bool = True):
    """Loads pickled files.

    Parameters
    ----------
    file: str
        The file to read. Must include the full file path.
        Including the .pickle extension is optional provided fix_extension is
        True.
    fix_extension: bool
        Whether to replace any existing extension with the '.pickle' file
        extension. Default is True.

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


class Savable:
    """Base class used for saving SklarPy objects."""
    _OBJ_NAME: str

    def __init__(self, name: str):
        """Base class used for saving SklarPy objects.

        Parameters
        ----------
        name : str
            The name of your object.
            When saving, if no file_path is specified,
            this will also be the name of the pickle file the object is saved
            to.
        """
        if name is None:
            name = self._OBJ_NAME
        elif not isinstance(name, str):
            raise TypeError("name must be a string")
        self._name: str = name

    @property
    def name(self) -> str:
        """The name of your object"""
        return self._name

    def save(self, file_path: str = None, overwrite: bool = False,
             fix_extension: bool = True) -> str:
        """Saves object as a pickled file.

        Parameters
        ----------
        file_path: Union[str, None]
            The location and file name where you are saving your object.
            If None, the object is saved under its name in the current working
            directory. If a file is given, it must include the full file path.
            The '.pickle' extension is optional provided fix_extension is True.
        overwrite: bool
            True to overwrite existing files saved under the same name.
            False to save under a unique name.
            Default is False.
        fix_extension: bool
            Whether to replace any existing extension with the '.pickle' file
            extension / add the '.pickle' extension if None given.
            Default is True.

        Returns
        -------
        file_name: str
            The path to the saved object

        See Also
        ---------
        sklarpy.load
        pickle
        dill
        """
        # argument checks
        if file_path is None:
            dir_path: str = os.getcwd()
            file_path = f'{dir_path}/{self.name}.pickle'
        elif not isinstance(file_path, str):
            raise TypeError("file argument must be a string.")

        for bool_arg in (overwrite, fix_extension):
            if not isinstance(bool_arg, bool):
                raise TypeError("overwrite, fix_extension arguments must both "
                                "be boolean.")

        # Changing file extension to .pickle
        file_name, extension = os.path.splitext(file_path)
        if fix_extension:
            extension = '.pickle'

        if not overwrite:
            # Saving under a unique file name
            count: int = 0
            unique_str: str = ''
            while Path(f'{file_name}{unique_str}{extension}').exists():
                count += 1
                unique_str = f' ({count})'
            file_name = f'{file_name}{unique_str}{extension}'

        # saving object
        try:
            with open(file_name, 'wb') as f:
                dill.dump(self, f)
            return file_name
        except Exception as e:
            raise SaveError(e)
