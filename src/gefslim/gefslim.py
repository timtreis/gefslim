from pathlib import Path

import anndata
import h5py
import numpy as np
import pandas as pd


class GEF:
    """Class that interacts with the .gef file."""

    def __init__(
        self,
        result_dir: str,
    ) -> None:
        self.result_dir = Path(result_dir)
        self.data = {}

    def get_counts_per_cell(self) -> pd.DataFrame:
        """Returns the number of probes per cell."""
        cellcut = self.read_cellcut().to_df()

        result = pd.DataFrame()
        result["counts"] = cellcut[["geneCount"]]
        result["cell_id"] = cellcut.index.tolist()

        return result

    def get_cell_borders(self, transformed: bool = True) -> pd.DataFrame:
        """Returns the spatial information per cell."""
        cellcut = self.read_cellcut().to_df()

        result = pd.DataFrame()
        result["cell_id"] = cellcut.index.tolist()
        result["x"] = cellcut["x"].values
        result["y"] = cellcut["y"].values
        result["border"] = cellcut["border"].values

        if not transformed:
            return result
        else:
            result["border"] = result.apply(
                lambda row: [[point[0] + row["x"], point[1] + row["y"]] for point in row["border"]], axis=1
            )
            del result["x"]
            del result["y"]

            return result

    def get_genecounts_per_cell(self) -> anndata.AnnData:
        """Returns the counts per gene per cell."""
        folder = self.result_dir / "041.cellcut"

        for file in folder.glob("*"):
            if "cellcut" in str(file):
                fname = file.name
                self.data[fname] = h5py.File(file)

        if len(self.data) > 1:
            raise NotImplementedError("Only 1 cellcut is supported for now")

        else:
            gene_data = self.data[next(iter(self.data))]["cellBin"]["gene"][:]
            geneExp_data = self.data[next(iter(self.data))]["cellBin"]["geneExp"][:]

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

    def read_cellcut(self) -> None:
        """Returns the cellcut file."""
        folder = self.result_dir / "041.cellcut"

        for file in folder.glob("*"):
            if "cellcut" in str(file):
                fname = file.name
                self.data[fname] = h5py.File(file)

        if len(self.data) > 1:
            raise NotImplementedError("Only 1 cellcut is supported for now")

        else:
            cell_data = self.data[next(iter(self.data))]["cellBin"]["cell"][:]
            cellBorder_data = self.data[next(iter(self.data))]["cellBin"]["cellBorder"][:]

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
