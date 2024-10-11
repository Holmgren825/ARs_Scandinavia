"""Preprocessing ARTMIP data before PCA analysis."""

import logging
import os
import re
from typing import Self, TypedDict, Union, Literal, Optional

import dask.array as da
import numpy as np
import xarray as xr
from ar_identify.feature import _get_labels
from ar_identify.utils import uniquify_id
from shapely.geometry import GeometryCollection, MultiPoint
from tqdm.autonotebook import tqdm

logger = logging.getLogger(__name__)


class PathDict(TypedDict):
    """Specify paths and filenames for a ArtmipDataset."""

    artmip_dir: str
    project_dir: str
    # region_ar_fname: str
    # What else is needed here?


class ArtmipDataset:
    """Streamlining the preprocessing of ARTMIP data for an AR PCA analysis."""

    def __init__(
        self: Self,
        path_dict: PathDict,
        region_shp: Optional[GeometryCollection] = None,
        overwrite: bool = False,
        n_year_batch: int = 1,
    ) -> None:
        # NOTE: We can't do an instance check on typed dicts.
        if isinstance(path_dict, dict):
            # We need to define a type for this dict and what it should look like.
            self.path_dict = path_dict
        else:
            raise TypeError("path_dict should be a dictionary")

        if region_shp is not None:
            if isinstance(region_shp, GeometryCollection):
                self.region_shp = region_shp
            else:
                raise TypeError("region_shp should be a GeometryCollection")

        if isinstance(n_year_batch, int):
            self.n_year_batch = n_year_batch

        self.ardt_name = self._extract_ardt_name()

        self.region_mask_ds: Optional[xr.DataArray] = None
        self.region_ar_ds: Optional[xr.DataArray] = None
        self.ar_tag_ds: Optional[xr.Dataset] = None
        self.ar_id_ds: Optional[xr.Dataset] = None
        self.overwrite = overwrite

    def _extract_ardt_name(self: Self) -> str:
        ardt_name = self.path_dict["artmip_dir"]
        ardt_name = os.path.basename(os.path.normpath(ardt_name))
        ardt_name = re.sub("ar", "ar_tag", ardt_name)
        return ardt_name

    def _create_ar_id_fname(self: Self, time_thin: int) -> str:
        sub = re.sub("1hr", f"{time_thin}hr", self.ardt_name)
        return re.sub("ar_tag", "ar_id", sub)

    def _get_timerange_str(self: Self, ds: xr.Dataset) -> str:
        first_timestep = ds.isel(time=0).time.dt.strftime("%Y%m%d").values
        last_timestep = ds.isel(time=-1).time.dt.strftime("%Y%m%d").values

        return f"{first_timestep}-{last_timestep}"

    def preprocess_artmip_catalog(self: Self) -> None:
        """Preprocess a single catalog of tier 2 artmip data."""
        # The first thing we want to do is to take the raw catalog, which contains a number of netcdf files,
        # and open these using xarray and save them as a chunked zarr store.
        ds = self._load_artmip_ds()
        # NOTE:Have to chunk the data since chunks are likely unequal,
        # which is not allowed by zarr.
        ds = ds.chunk("auto")
        timerange_str = self._get_timerange_str(ds)
        logger.info("Saving to zarr store.")
        store_path = (
            os.path.join(self.path_dict["artmip_dir"], self.ardt_name)
            + timerange_str
            + ".zarr"
        )
        if self.overwrite or not os.path.exists(store_path):
            if self.overwrite:
                mode: Union[Literal["w"] | None] = "w"
                logger.info(f"Overwriting store {store_path}")
            else:
                mode = None
            ds.to_zarr(store_path, mode=mode)
        else:
            logger.info(f"Store {store_path} already exist.")
        logger.info("Done")
        self.ar_tag_ds = xr.open_zarr(store_path)

    def _load_artmip_ds(self: Self) -> xr.Dataset:
        artmip_files = os.path.join(self.path_dict["artmip_dir"], self.ardt_name)
        artmip_files = artmip_files + ".*.nc"
        raw_ds = xr.open_mfdataset(artmip_files)
        if not list(raw_ds.variables.keys()) == [
            "ar_binary_tag",
            "lat",
            "lon",
            "time",
        ]:
            raise ValueError(
                "The originial artmip catalog does not contain the correct variables."
            )
        return raw_ds

    def get_unique_ar_ids(
        self: Self,
        show_progress: bool = False,
        first_year: Union[int | None] = None,
        last_year: Union[int | None] = None,
        time_thin: int = 6,
    ) -> None:
        """Get AR feature ids from ARTMIP data."""
        # NOTE: This was done through a separate script, but should be a part of this pipeline now.

        if self.ar_tag_ds is None:
            raise AttributeError(
                "ar_tag_ds has not be initiated for the ArtmipDataset, run the preprocessor."
            )
        tag_ds = self.ar_tag_ds
        tag_ds = tag_ds.thin({"time": time_thin})

        timerange_str = self._get_timerange_str(tag_ds)
        ar_id_fname = self._create_ar_id_fname(time_thin)
        store_path = (
            os.path.join(self.path_dict["artmip_dir"], ar_id_fname)
            + timerange_str
            + ".zarr"
        )

        if first_year is None:
            first_year = tag_ds.time.dt.year.values[0]
        if last_year is None:
            last_year = tag_ds.time.dt.year.values[-1]

        # Structure
        # TODO: Shoudl this be static?
        # I think so, why would it change?
        structure = np.array(
            [np.zeros((3, 3)), np.ones((3, 3)), np.zeros((3, 3))], dtype=bool
        )

        logger.info("Beginning generating unique ids for ar objects")
        # NOTE: We should only enter this loop if store does not exist, or we are overwriting
        if self.overwrite or not os.path.exists(store_path):
            # +1 since range upper bound is exclusive.
            for i, year in tqdm(
                enumerate(range(first_year, last_year + 1, self.n_year_batch)),
                disable=not show_progress,
            ):
                tag_ds_sel = tag_ds.sel(
                    time=slice(f"{year}", f"{year+self.n_year_batch-1}")
                )

                features = _get_labels(
                    tag_ds_sel.ar_binary_tag, wrap_axes=(2,), structure=structure
                )
                if isinstance(features, xr.DataArray):
                    features = features.rechunk((1, -1, -1))

                features = da.where(
                    features > 0,
                    uniquify_id(
                        tag_ds_sel["time.year"].values.reshape((-1, 1, 1)), features
                    ),
                    0,
                )

                ds_feat = xr.DataArray(
                    features,
                    dims=("time", "lat", "lon"),
                    coords={
                        "time": tag_ds_sel.time,
                        "lat": tag_ds_sel.lat,
                        "lon": tag_ds_sel.lon,
                    },
                    name="ar_unique_id",
                )

                logger.info(f"Compute and save {year}...")
                # NOTE: if i = 0, it means that we are on the first year of the dataset
                if not i:
                    mode = "w" if self.overwrite else None
                    ds_feat.to_zarr(store_path, mode)

                else:
                    ds_feat.to_zarr(store_path, append_dim="time")
        else:
            logger.info(f"Store {store_path} already exists.")
        logger.info("Unique ids done")

        self.ar_id_ds = xr.open_zarr(store_path)

    def create_region_mask(self: Self) -> None:
        """Create a binary mask for the specified region used to extract atmospheric rivers."""
        logger.info("Start: Create region mask.")
        if self.region_shp is None:
            raise AttributeError("region_shp not set for ArtmipDataset")
        if self.ar_id_ds is None:
            raise AttributeError("ar_id_ds is not initiated for ArtmipDataset")

        x, y = np.meshgrid(self.ar_id_ds.lon, self.ar_id_ds.lat)

        x_flat = x.flatten()
        y_flat = y.flatten()

        lon_lat_points = np.vstack([x_flat, y_flat])
        points = MultiPoint(lon_lat_points.T)

        indices = [i for i, p in enumerate(points.geoms) if self.region_shp.contains(p)]

        # Create the mask
        mask = np.ones(self.ar_id_ds.ar_unique_id.shape[1:], dtype=bool)
        # Set values within the specified region to false, e.g. the areas we want to keep.
        mask[np.unravel_index(indices, mask.shape)] = False
        self.region_mask_ds = xr.DataArray(
            data=~mask,
            dims=["lat", "lon"],
            coords={"lon": self.ar_id_ds.lon, "lat": self.ar_id_ds.lat},
        )

    def select_region_ars(self: Self) -> None:
        """Select ID numbers of ARs that intersect the specified region (region_mask)."""
        if self.ar_id_ds is None:
            raise AttributeError("ar_id_ds is not initiated for ArtmipDataset")

        ids = da.unique(
            self.ar_id_ds.ar_features.where(self.region_mask_ds, np.nan).data
        )
        ids = ids.compute()
        mask_isin = da.isin(self.ar_id_ds.ar_feautures.data, ids[1:-1])
        region_ars = self.ar_id_ds.ar_feautures.where(mask_isin, np.nan)
        region_ars.name = "ar_features"
        self.region_ar_ds = region_ars

    def _extract_fname(self: Self) -> str:
        raise NotImplementedError
