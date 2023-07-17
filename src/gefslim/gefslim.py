from pathlib import Path

import anndata
import numpy as np
import pandas as pd

from gefslim.utils import _get_h5_from_gef


class GEF:
    """
    A class that provides methods for interacting with .gef files.

    Attributes
    ----------
    result_dir : str
        The directory where the .gef files are located.
    data : dict
        A dictionary that stores the loaded data from the .gef files.
    """

    def __init__(
        self,
        result_dir: str,
    ) -> None:
        """
        Initialize a GEF instance.

        Parameters
        ----------
        result_dir : str
            The directory where the .gef files are located.
        """
        self.result_dir = Path(result_dir)
        self.data = {}

    def _read_cellcut_to_df(self, name: str) -> pd.DataFrame:
        """
        Convert a cellcut file to a pandas DataFrame.

        Parameters
        ----------
        name : str
            The name of the cellcut file to be read.

        Returns
        -------
        pd.DataFrame
            A DataFrame representation of the cellcut data with an
            additional column "cell_id" for the original indices.
        """
        cellcut_df = self.read_cellcut(name=name).to_df()
        cellcut_df["cell_id"] = cellcut_df.index
        return cellcut_df

    def get_counts_per_cell(self, name: str) -> pd.DataFrame:
        """
        Retrieve the number of probes per cell from a cellcut file.

        Parameters
        ----------
        name : str
            The name of the cellcut file.

        Returns
        -------
        pd.DataFrame
            A DataFrame with 'cell_id' and 'counts' columns, where
            'counts' represents the number of probes per cell. Returns
            an empty DataFrame if an error occurs during the read operation.
        """
        cellcut = self._read_cellcut_to_df(name)
        return cellcut[["cell_id", "geneCount"]].rename(columns={"geneCount": "counts"})

    def get_area_per_cell(self, name: str) -> pd.DataFrame:
        """
        Retrieve the size in pixels of each cell from a cellcut file.

        Parameters
        ----------
        name : str
            The name of the cellcut file.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the 'cell_id' and 'area' for each cell.
        """
        cellcut = self._read_cellcut_to_df(name)
        return cellcut[["cell_id", "area"]]

    def get_cell_borders(self, name: str, transformed: bool = True) -> pd.DataFrame:
        """
        Extract the spatial information per cell from a cellcut file.

        Parameters
        ----------
        name : str
            The name of the cellcut file.
        transformed : bool, optional
            Whether or not to transform the cell's border data. If True,
            each point in the border data will have the cell's x and y
            values added to it. Default is True.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the cell's ID and its border data. If
            'transformed' is False, the cell's x and y values are also included.
        """
        cellcut = self._read_cellcut_to_df(name)

        if transformed:
            cellcut["border"] = cellcut.apply(
                lambda row: [[point[0] + row["x"], point[1] + row["y"]] for point in row["border"]], axis=1
            )
            return cellcut[["cell_id", "border"]]

        return cellcut[["cell_id", "x", "y", "border"]]

    def get_genecounts_per_cell(self, name: str) -> pd.DataFrame:
        """
        Retrieve the counts per gene per cell.

        Parameters
        ----------
        name : str
            The name of the .gef file.

        Returns
        -------
        pd.DataFrame
            A DataFrame sorted by 'cell_id' and 'gene', containing genes,
            cell IDs and corresponding gene counts. Columns are 'gene',
            'cell_id' and 'counts'.
        """
        h5 = _get_h5_from_gef(
            result_path=self.result_dir,
            folder_name="041.cellcut",
            file_name=name,
        )

        gene_info = h5["cellBin"]["gene"][:]
        gene_expression_info = h5["cellBin"]["geneExp"][:]

        cell_ids = gene_expression_info["cellID"]
        counts = gene_expression_info["count"]

        # Extract which probes were found in which cells
        probe_df = pd.DataFrame()
        probe_df["genes"] = [
            name.decode() for name, cell_count in gene_info[["geneName", "cellCount"]] for _ in range(cell_count)
        ]
        probe_df["cellID"] = cell_ids
        probe_df["counts"] = counts

        return (
            probe_df.rename(columns={"genes": "gene", "cellID": "cell_id"})
            .sort_values(["cell_id", "gene"])
            .reset_index(drop=True)
        )

    def read_cellcut(self, name: str) -> anndata.AnnData:
        """
        Read a cellcut file and return an AnnData object.

        Parameters
        ----------
        name : str
            The name of the cellcut file.

        Returns
        -------
        anndata.AnnData
            An AnnData object containing cellcut data. The `.obs_names` attribute
            is set to the 'id' field from the cellcut data, and the `.var_names`
            attribute is set to the names of the cellcut data fields.
        """
        # Get the h5 file from the gef result directory
        h5 = _get_h5_from_gef(
            result_path=self.result_dir,
            folder_name="041.cellcut",
            file_name=name,
        )

        cell_data = h5["cellBin"]["cell"][:]
        cell_border_data = h5["cellBin"]["cellBorder"][:]

        # Prepare a dictionary to create DataFrame
        cellcut_data_dict = {name: cell_data[name] for name in cell_data.dtype.names if name != "id"}

        cellcut_data_dict["border"] = [border[~np.all(border == [32767, 32767], axis=1)] for border in cell_border_data]
        cellcut_df = pd.DataFrame(cellcut_data_dict)
        cellcut_df.index = cell_data["id"].astype(str)

        return anndata.AnnData(cellcut_df)

    def get_genecounts_per_spot(self, name: str, binsize: int = 100) -> pd.DataFrame:
        """
        Calculate gene counts per specified bin size around a spot (x/y).

        Parameters
        ----------
        name : str
            The name of the gene expression file.
        binsize : int, optional
            The size of the bin for which the gene counts are calculated. Defaults to 100.

        Returns
        -------
        pd.DataFrame
            DataFrame containing x, y coordinates, gene, and corresponding counts,
            sorted by x, y and gene.
        """
        h5 = _get_h5_from_gef(
            result_path=self.result_dir,
            folder_name="04.tissuecut",
            file_name=name,
        )

        exp_data = h5["geneExp"][f"bin{binsize}"]["expression"][:]
        gene_data = h5["geneExp"][f"bin{binsize}"]["gene"][:]

        result = pd.DataFrame()
        result["x"] = exp_data["x"]
        result["y"] = exp_data["y"]
        result["gene"] = [name.decode() for name, cell_count in gene_data[["gene", "count"]] for _ in range(cell_count)]
        result["counts"] = exp_data["count"]

        return result.sort_values(["x", "y", "gene"]).reset_index(drop=True)

    def get_gene_stats(self, name: str) -> pd.DataFrame:
        """
        Retrieve gene statistics from a HDF5 file and return it as a sorted DataFrame.

        Parameters
        ----------
        name : str
            Name of the gene file.

        Returns
        -------
        pd.DataFrame
            DataFrame containing gene statistics. Columns include 'gene', 'MIDcount',
            and 'E10'. The DataFrame is sorted in descending order by 'MIDcount' column
            and the index is reset.
        """
        h5 = _get_h5_from_gef(
            result_path=self.result_dir,
            folder_name="04.tissuecut",
            file_name=name,
        )

        gene_stat_data = h5["stat"]["gene"][:]

        gene_stats_df = pd.DataFrame(
            {
                "gene": [gene.decode() for gene in gene_stat_data["gene"]],
                "MIDcount": gene_stat_data["MIDcount"],
                "E10": gene_stat_data["E10"],
            }
        )

        return gene_stats_df.sort_values(by="MIDcount", ascending=False).reset_index(drop=True)
