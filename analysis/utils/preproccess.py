"""Preprocessing ARTMIP data before PCA analysis."""

import logging
import os
import re
from itertools import repeat
from typing import (
    Hashable,
    Literal,
    Mapping,
    Optional,
    Self,
    TypedDict,
    Union,
)

import dask.array as da
import numpy as np
import xarray as xr
import pandas as pd
from ar_identify.feature import _get_labels
from ar_identify.utils import uniquify_id
from numpy.typing import NDArray
from shapely.geometry import MultiPoint
from shapely.geometry.base import BaseGeometry
from tqdm.autonotebook import tqdm

logger = logging.getLogger(__name__)


class PathDict(TypedDict):
    """Specify paths and filenames for a ArtmipDataset."""

    artmip_dir: str
    project_dir: str
    region_name: str
    # What else is needed here?


class ArtmipDataset:
    """Streamlining the preprocessing of ARTMIP data for an AR PCA analysis."""

    def __init__(
        self: Self,
        path_dict: PathDict,
        region_shp: Optional[BaseGeometry] = None,
        overwrite: bool = False,
        n_batch_chunks: int = 20,
        time_thin: Optional[int] = None,
    ) -> None:
        # NOTE: We can't do an instance check on typed dicts.
        if isinstance(path_dict, dict):
            # We need to define a type for this dict and what it should look like.
            self.path_dict = path_dict
        else:
            raise TypeError("path_dict should be a dictionary")

        if region_shp is not None:
            if isinstance(region_shp, BaseGeometry):
                self.region_shp = region_shp
            else:
                raise TypeError("region_shp should be a BaseGeometry")

        if isinstance(n_batch_chunks, int):
            self.n_batch_chunks = n_batch_chunks
        if isinstance(time_thin, int):
            self.time_thin: Optional[int] = time_thin
        else:
            self.time_thin = None

        self.ardt_raw_name = self._extract_ardt_raw_name()
        self.ardt_name = self._extract_ardt_name()
        self.ardt_proj_dir = os.path.join(
            self.path_dict["project_dir"], self._extract_ardt_dir()
        )
        if not os.path.exists(self.ardt_proj_dir):
            os.makedirs(self.ardt_proj_dir)

        self.region_mask_ds: Optional[xr.DataArray] = None
        self.region_ar_ds: Optional[xr.DataArray] = None
        self.ar_tag_ds: Optional[xr.Dataset] = None
        self.ar_id_ds: Optional[xr.Dataset] = None
        self.overwrite = overwrite

    def _extract_ardt_raw_name(self: Self) -> str:
        name = os.path.basename(os.path.normpath(self.path_dict["artmip_dir"]))
        return name

    def _extract_ardt_name(self: Self) -> str:
        name = self._extract_ardt_raw_name()
        return re.sub("ar", "ar_tag", name)

    def _extract_ardt_dir(self: Self) -> str:
        name = self._extract_ardt_raw_name()
        name = re.sub(".1hr", "", name)
        name = re.sub(".ar", "", name)
        return name

    def _get_timerange_str(self: Self, ds: xr.Dataset) -> str:
        first_timestep = ds.isel(time=0).time.dt.strftime("%Y%m%d").values
        last_timestep = ds.isel(time=-1).time.dt.strftime("%Y%m%d").values

        return f"{first_timestep}-{last_timestep}"

    def multi_subs(self: Self, name: str, subs: dict[str, str]) -> str:
        """Perform any number of regex substitutions on name."""
        for key, value in subs.items():
            name = re.sub(key, value, name)
        return name

    def _generate_iter_blocks(
        self: Self,
        chunks: Mapping[Hashable, tuple[int, ...]],
        n_batch: Optional[int] = None,
    ) -> Mapping[str, NDArray]:
        if self.ar_tag_ds is None:
            raise AttributeError("ar_tag_ds not initiatetd for ArtmipDataset")
        time_chunks = chunks["time"]
        lat_chunks = chunks["lat"]
        lon_chunks = chunks["lon"]
        # TODO: We assume we only have chunks along one dimenison for now.
        if len(lat_chunks) > len(time_chunks) or len(lon_chunks) > len(time_chunks):
            raise ValueError(
                "Suspected chunking along other dimension than time. Currently not supported."
            )
        if n_batch is None and self.n_batch_chunks is None:
            raise ValueError(
                "Please provide n_batch or set the n_batch attribute for the ArtmipDataset."
            )
        n_batch = self.n_batch_chunks if n_batch is None else n_batch

        time_chunks_ar = np.zeros(len(time_chunks) + 1, dtype=int)
        lat_chunks_ar = np.zeros((len(lat_chunks) + 1), dtype=int)
        lon_chunks_ar = np.zeros((len(lon_chunks) + 1), dtype=int)

        time_chunks_ar[1:] += np.cumsum(time_chunks)
        lat_chunks_ar[1:] += np.cumsum(lat_chunks)
        lon_chunks_ar[1:] += np.cumsum(lon_chunks)

        # See todo above.
        # TODO: How can we batch this? E.g multiple pariwise. Doing things block is slow.
        time_chunk_pairs = np.asarray(
            np.lib.stride_tricks.sliding_window_view(time_chunks_ar, n_batch)
        )[:, [0, -1]][:: n_batch - 1]
        if not time_chunk_pairs[-1, -1] == time_chunks_ar[-1]:
            time_chunk_pairs = np.concat(
                (time_chunk_pairs, [[time_chunk_pairs[-1][-1], time_chunks_ar[-1]]])
            )
        iter_blocks = {
            "time": time_chunk_pairs,
            "lat": np.asarray(list(repeat(lat_chunks_ar, len(time_chunk_pairs)))),
            "lon": np.asarray(list(repeat(lon_chunks_ar, len(time_chunk_pairs)))),
        }

        return iter_blocks

    def preprocess_artmip_catalog(self: Self) -> None:
        """Preprocess a single catalog of tier 2 artmip data."""
        # The first thing we want to do is to take the raw catalog, which contains a number of netcdf files,
        # and open these using xarray and save them as a chunked zarr store.
        if self.time_thin is None:
            raise AttributeError(
                "Attribute time_thin of ArtmipDataset is None, should be int."
            )

        ds_raw = self._load_artmip_ds()
        # NOTE:Have to chunk the data since chunks are likely unequal,
        # which is not allowed by zarr.
        # We should chunk this as if it was int64, not int8.
        ds = ds_raw.thin({"time": self.time_thin})
        preferred_chunksizes = ds.astype("int64").chunk("auto").chunksizes
        ds = ds.chunk(preferred_chunksizes)

        timerange_str = self._get_timerange_str(ds)
        logger.info("Saving to zarr store.")
        fname = self.multi_subs(self.ardt_raw_name, {"1hr": f"{self.time_thin}hr"})
        fname = self.generate_filename([fname, timerange_str])
        store_path = os.path.join(self.path_dict["artmip_dir"], fname) + ".zarr"
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
        first_year: Optional[int] = None,
        last_year: Optional[int] = None,
    ) -> None:
        """Get AR feature ids from ARTMIP data."""
        # NOTE: This was done through a separate script, but should be a part of this pipeline now.

        if self.ar_tag_ds is None:
            raise AttributeError(
                "ar_tag_ds has not be initiated for the ArtmipDataset, run the preprocessor."
            )

        tag_ds = self.ar_tag_ds

        timerange_str = self._get_timerange_str(tag_ds)
        ar_id_fname = self.multi_subs(
            self.ardt_raw_name, {"1hr": f"{self.time_thin}hr", "ar": "ar_id"}
        )
        fname = self.generate_filename([ar_id_fname, timerange_str])
        store_path = os.path.join(self.ardt_proj_dir, fname) + ".zarr"

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
            iter_blocks = self._generate_iter_blocks(self.ar_tag_ds.chunksizes)
            for i, (time_slice, lat_slice, lon_slice) in tqdm(
                enumerate(
                    zip(iter_blocks["time"], iter_blocks["lat"], iter_blocks["lon"])
                ),
                disable=not show_progress,
            ):
                tag_ds_sel = tag_ds.isel(
                    time=slice(*time_slice),
                    lat=slice(*lat_slice),
                    lon=slice(*lon_slice),
                )

                features = _get_labels(
                    tag_ds_sel.ar_binary_tag, wrap_axes=(2,), structure=structure
                )
                if isinstance(features, xr.DataArray):
                    features = features.rechunk((1, -1, -1))

                id_uniqify = tag_ds_sel["time.year"].values.reshape((-1, 1, 1))
                id_uniqify += i * 1000
                features = da.where(
                    features > 0,
                    uniquify_id(id_uniqify, features),
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

                logger.info(f"Compute and save block {i}...")
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
        logger.info("End: Create region mask.")

    def select_region_ars(self: Self) -> None:
        """Select ID numbers of ARs that intersect the specified region (region_mask)."""
        if self.ar_id_ds is None:
            raise AttributeError("ar_id_ds is not initiated for ArtmipDataset")
        if self.region_mask_ds is None:
            raise AttributeError("region_mask_ds is not initiated for ArtmipDataset")
        if self.time_thin is None:
            raise AttributeError(
                "Attribute time_thin of ArtmipDataset is None, should be int."
            )
        logger.info("Start: Select region ARs.")

        timerange_str = self._get_timerange_str(self.ar_id_ds)
        fname = self.multi_subs(
            self.ardt_raw_name, {"1hr": f"{self.time_thin}hr", "ar": "ar_id"}
        )
        fname = self.generate_filename(
            [fname, self.path_dict["region_name"], timerange_str]
        )

        store_path = os.path.join(self.ardt_proj_dir, fname) + ".zarr"
        if self.overwrite or not os.path.exists(store_path):
            if self.overwrite:
                mode: Union[Literal["w"] | None] = "w"
                logger.info(f"Will overwrite store {store_path}")
            else:
                mode = None

            ids = da.unique(
                self.ar_id_ds.ar_unique_id.where(self.region_mask_ds, np.nan).data
            )
            logger.info("Computing unique region AR ids.")
            ids = ids.compute()
            logger.info("DONE: Computing unique region AR ids.")
            mask_isin = da.isin(self.ar_id_ds.ar_unique_id.data, ids[1:-1])
            region_ars = self.ar_id_ds.ar_unique_id.where(mask_isin, np.nan)
            logger.info("Saving to zarr store.")
            region_ars.to_zarr(store_path, mode=mode)
        else:
            logger.info(f"Store {store_path} already exist.")

        self.region_ar_ds = xr.open_zarr(store_path)
        logger.info("End: Select region ARs.")

    def generate_filename(
        self: Self, fname_params: list[str], sep: Literal[".", "-", "_"] = "."
    ) -> str:
        """Generate a filename from fname_params separated using sep."""
        return sep.join(fname_params)
