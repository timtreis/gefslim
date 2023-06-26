from pathlib import Path

import anndata
import numpy as np
import pandas as pd

from gefslim.utils import _get_h5_from_gef


class GEF:
    """Class that interacts with the .gef file."""

    def __init__(
        self,
        result_dir: str,
    ) -> None:
        """
        Initialize GEF class.

        Args:
            result_dir (str): Directory where results are located.
        """
        self.result_dir = Path(result_dir)
        self.data = {}

    def _read_cellcut_to_df(self, name: str) -> pd.DataFrame:
        """
        Read cellcut and return as DataFrame.

        Args:
            name (str): File name.

        Returns
        -------
            pd.DataFrame: DataFrame of cellcut data.
        """
        cellcut = self.read_cellcut(name=name).to_df()
        cellcut["cell_id"] = cellcut.index.tolist()
        return cellcut

    def get_counts_per_cell(self, name: str) -> pd.DataFrame:
        """
        Returns the number of probes per cell.

        Args:
            name (str): File name.

        Returns
        -------
            pd.DataFrame: DataFrame with cell_id and count of probes.
        """
        cellcut = self._read_cellcut_to_df(name)
        return cellcut[["cell_id", "geneCount"]].rename(columns={"geneCount": "counts"})

    def get_area_per_cell(self, name: str) -> pd.DataFrame:
        """
        Returns the size in pixel per cell.

        Args:
            name (str): File name.

        Returns
        -------
            pd.DataFrame: DataFrame with cell_id and area.
        """
        cellcut = self._read_cellcut_to_df(name)
        return cellcut[["cell_id", "area"]]

    def get_cell_borders(self, name: str, transformed: bool = True) -> pd.DataFrame:
        """
        Returns the spatial information per cell.

        Args:
            name (str): File name.
            transformed (bool, optional): If true, returns borders transformed. Defaults to True.

        Returns
        -------
            pd.DataFrame: DataFrame with cell_id and borders (and x,y if not transformed).
        """
        cellcut = self._read_cellcut_to_df(name)

        if transformed:
            cellcut["border"] = cellcut.apply(
                lambda row: [[point[0] + row["x"], point[1] + row["y"]] for point in row["border"]]
                if transformed
                else row["border"],
                axis=1,
            )
            return cellcut[["cell_id", "border"]]
        else:
            return cellcut[["cell_id", "x", "y", "border"]]

    def get_genecounts_per_cell(self, name: str) -> anndata.AnnData:
        """Returns the counts per gene per cell."""
        h5 = _get_h5_from_gef(
            result_path=self.result_dir,
            folder_name="041.cellcut",
            file_name=name,
        )

        gene_data = h5["cellBin"]["gene"][:]
        geneExp_data = h5["cellBin"]["geneExp"][:]

        # Extract which probes were found in which cells
        probe_df = pd.DataFrame()
        gene_list = [
            name.decode() for name, cell_count in gene_data[["geneName", "cellCount"]] for _ in range(cell_count)
        ]
        probe_df["genes"] = gene_list
        probe_df["cellID"] = geneExp_data["cellID"]
        probe_df["counts"] = geneExp_data["count"]

        probe_df = probe_df.rename(columns={"genes": "gene", "cellID": "cell_id"})

        return probe_df.sort_values(["cell_id", "gene"]).reset_index(drop=True)

    def read_cellcut(self, name: str) -> anndata.AnnData:
        """Returns the cellcut file."""
        h5 = _get_h5_from_gef(
            result_path=self.result_dir,
            folder_name="041.cellcut",
            file_name=name,
        )

        cell_data = h5["cellBin"]["cell"][:]
        cellBorder_data = h5["cellBin"]["cellBorder"][:]

        tmp = {}
        names = list(cell_data.dtype.names)
        names.remove("id")
        for name in names:
            tmp[name] = cell_data[name]

        cellcut_df = pd.DataFrame(tmp, columns=names)

        # truncate border points
        cellcut_df["border"] = [border[~np.all(border == [32767, 32767], axis=1)] for border in cellBorder_data]
        names += ["border"]

        cellcut_adata = anndata.AnnData(X=cellcut_df.values)
        cellcut_adata.obs_names = [str(i) for i in cell_data["id"]]
        cellcut_adata.var_names = names

        return cellcut_adata

    def get_genecounts_per_spot(self, name: str, binsize: int = 100) -> pd.DataFrame:
        """Returns the counts per bin around (x/y)."""
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
        """Returns the counts per bin around (x/y)."""
        h5 = _get_h5_from_gef(
            result_path=self.result_dir,
            folder_name="04.tissuecut",
            file_name=name,
        )

        stat_data = h5["stat"]["gene"][:]

        result = pd.DataFrame()
        result["gene"] = [gene.decode() for gene in stat_data["gene"]]
        result["MIDcount"] = stat_data["MIDcount"]
        result["E10"] = stat_data["E10"]

        return result.sort_values("MIDcount", ascending=False).reset_index(drop=True)
