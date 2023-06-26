import os
import pathlib

import h5py


def _get_h5_from_gef(
    result_path: pathlib.Path,
    folder_name: str,
    file_name: str,
) -> h5py.File:
    """Returns the connection to a h5py file."""
    if not isinstance(result_path, pathlib.Path):
        raise TypeError(f"Parameter 'result_path' must be a pathlib.Path, got: {type(result_path)}")

    if not isinstance(folder_name, str):
        raise TypeError(f"Parameter 'folder_name' must be a string, got: {type(folder_name)}")

    if not isinstance(file_name, str):
        raise TypeError(f"Parameter 'file_name' must be a string, got: {type(file_name)}")

    if ".gef" not in file_name:
        raise ValueError("Parameter 'file_name' must be a `.gef` file.")

    path = result_path / folder_name / file_name
    if not os.path.exists(path):
        raise FileNotFoundError(f"File '{path}' does not exist.")

    return h5py.File(path, "r")
