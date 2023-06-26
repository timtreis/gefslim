import pathlib

import h5py


def _get_h5_from_gef(
    result_path: pathlib.Path,
    folder_name: str,
    file_name: str,
) -> h5py.File:
    """
    Returns the connection to an h5py file given its location details.

    Args:
        result_path (pathlib.Path): The base directory of the file.
        folder_name (str): The subdirectory (folder) of the file.
            The actual file path is thus: result_path/folder_name/file_name
        file_name (str): The name of the file with the extension.
            Should have a `.gef` extension.

    Raises
    ------
        TypeError: If the provided arguments are not of the correct type.
        ValueError: If the `file_name` does not have a `.gef` extension.
        FileNotFoundError: If the file does not exist at the specified path.

    Returns
    -------
        h5py.File: The h5py file object for the requested file in read mode.
    """
    # Check argument types
    if not isinstance(result_path, pathlib.Path):
        raise TypeError(f"Parameter 'result_path' must be a pathlib.Path, got: {type(result_path)}")

    if not isinstance(folder_name, str):
        raise TypeError(f"Parameter 'folder_name' must be a string, got: {type(folder_name)}")

    if not isinstance(file_name, str):
        raise TypeError(f"Parameter 'file_name' must be a string, got: {type(file_name)}")

    # Check file extension
    if not file_name.endswith(".gef"):
        raise ValueError("Parameter 'file_name' must be a `.gef` file.")

    # Construct path and check existence
    path = result_path / folder_name / file_name
    if not path.exists():
        raise FileNotFoundError(f"File '{path}' does not exist.")

    # Return the h5py file object
    return h5py.File(str(path), "r")
