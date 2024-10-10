"""Some utlities for doing a PCA analysis on atmospheric rivers and precipitation."""

from typing import Optional, Literal

import os
from typing import Self, Sequence, Union

import xarray as xr
import xeofs as xe  # type: ignore
from xeofs.single.base_model_single_set import BaseModel  # type: ignore


class ComputePCA:
    """Lorem ipsum."""

    def __init__(
        self: Self,
        data: Union[xr.DataArray | list[xr.DataArray]],
        model: BaseModel,
        base_path: str,
        result_fnames: dict,
        combined_pca: bool = False,
        normalize: bool = True,
        sample_dim: str = "time",
    ) -> None:
        """"""
        if isinstance(data, xr.DataArray) or isinstance(data, list):
            self.data = data
        else:
            raise TypeError("Data should be an xr.DataArray, or a sequence of.")

        if isinstance(model, BaseModel):
            self.model = model
        else:
            raise TypeError("Model is not a xeofs model.")

        if isinstance(sample_dim, str):
            self.sample_dim = sample_dim
        else:
            raise TypeError("sample_dim should be a string")
        if isinstance(combined_pca, bool):
            self.combined_pca = combined_pca
        else:
            raise TypeError("combined_pca should be a bool")

        if isinstance(normalize, bool):
            self.normalize = normalize
        else:
            raise TypeError("normalize should be a bool")

        if isinstance(base_path, str):
            if os.path.exists(base_path):
                self.base_path = base_path
            else:
                raise ValueError("base_path does not exist.")
        else:
            raise TypeError("base_path shoud be a string.")

        if isinstance(result_fnames, dict):
            self.result_fnames = result_fnames

        # Where to store the EOF results.
        self.eofs_component_list: list[xr.Dataset] = []
        self.eofs_explained_variance_list: list[xr.Dataset] = []
        self.eofs_scores_list: list[xr.Dataset] = []

        self.pca_results: list[list[xr.Dataset]] = [
            self.eofs_component_list,
            self.eofs_explained_variance_list,
            self.eofs_scores_list,
        ]

        if self.normalize:
            self.normalize_data()

    def fit(self: Self) -> None:
        """Perform the PCA analysis and PCA rotation."""
        # Fit the inital model
        self._fit_helper(self.model, X=self.data, dim=self.sample_dim)

        # Fit rotation models
        rot_var = xe.single.EOFRotator(n_modes=self.model.n_modes, power=1)
        rot_pro = xe.single.EOFRotator(n_modes=self.model.n_modes, power=4)

        self._fit_helper(rot_var, model=self.model)
        self._fit_helper(rot_pro, model=self.model)

    def _fit_helper(self: Self, base_model: BaseModel, **kwargs) -> None:
        # Here we fit the specific model and append its results to the object.
        base_model.fit(**kwargs)

        self.eofs_component_list.append(base_model.components())
        self.eofs_explained_variance_list.append(base_model.explained_variance_ratio())
        self.eofs_scores_list.append(base_model.scores())

    def normalize_data(self: Self) -> None:
        """Normalize the datasets before the PCA analysis."""
        if isinstance(self.data, Sequence):
            for i, dataset in enumerate(self.data):
                self.data[i] = self.min_max_norm(dataset)
        else:
            self.data = self.min_max_norm(self.data)

    def min_max_norm(self: Self, dataset: xr.DataArray) -> xr.DataArray:
        """Min-max normalizer."""
        return (dataset - dataset.min()) / (dataset.max() - dataset.min())

    def save(self: Self, mode: Optional[Literal["w"]] = None) -> None:
        """Save the results of the PCA analysis."""
        for dataset_list, fname in zip(
            self.pca_results, ["comp_name", "exp_var_name", "scores_name"]
        ):
            dataset = self.result_to_ds(dataset_list, fname)
            path = os.path.join(self.base_path, self.result_fnames[fname])
            dataset.to_zarr(path, mode=mode)

    def result_to_ds(self: Self, dataset: list, fname: str) -> xr.Dataset:
        """Convert list of PCA results to a single xarray dataset."""
        if self.combined_pca and fname == "comp_name":
            return combined_eof_result_list_to_ds(dataset)
        else:
            return eof_result_list_to_ds(dataset)


def load_pca_results(base_path: str, fnames: dict) -> Sequence[xr.Dataset]:
    """Load AR PCA results."""
    datasets = []
    for key in fnames.values():
        datasets.append(xr.open_dataset(os.path.join(base_path, key), engine="zarr"))

    return datasets


def eof_result_list_to_ds(result_list: list) -> xr.Dataset:
    """Convert a EOF result list to a single dataset by adding a coordinate for the rotation."""
    for i, rot in enumerate([0, 1, 4]):
        result_list[i] = result_list[i].assign_coords({"rotation": rot})
    result_ds = xr.concat(result_list, dim="rotation")
    return result_ds


def combined_eof_result_list_to_ds(result_list: list) -> xr.Dataset:
    """Convert a Combined EOF result list to a single dataset by adding a coordinate for the rotation."""
    for i, rot in enumerate([0, 1, 4]):
        # TODO: This should probably not be hard coded.
        for j, var in enumerate(["ar", "tp"]):
            result_list[i][j] = result_list[i][j].assign_coords(
                {"rotation": rot, "variable": var}
            )
        try:
            result_list[i][1] = result_list[i][1].rename(
                {"latitude": "lat", "longitude": "lon"}
            )
        except ValueError as e:
            if (
                str(e)
                == "cannot rename 'latitude' because it is not a variable or dimension in this dataset"
            ):
                pass
            else:
                raise
    result_list_ds = xr.combine_nested(result_list, concat_dim=["rotation", "variable"])
    return result_list_ds
