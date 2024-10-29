"""Some utilities."""

import logging
from typing import overload

import numpy as np
import xarray as xr
from distributed.client import Client
from xr_utils.coordinates import conv_lon_to_180

logger = logging.getLogger(__name__)


@overload
def subsel_ds(
    ds: xr.DataArray,
    lat_slice: tuple[int, int],
    lon_slice: tuple[int, int],
    start_year: str,
    end_year: str,
) -> xr.DataArray: ...


@overload
def subsel_ds(
    ds: xr.Dataset,
    lat_slice: tuple[int, int],
    lon_slice: tuple[int, int],
    start_year: str,
    end_year: str,
) -> xr.Dataset: ...


def subsel_ds(
    ds: xr.DataArray | xr.Dataset,
    lat_slice: tuple[int, int],
    lon_slice: tuple[int, int],
    start_year: str,
    end_year: str,
) -> xr.DataArray | xr.Dataset:
    """Generate a subselection of given dataset."""
    if ds.cf["longitude"].values[-1] > 359:
        logger.info("0-360 coordinates detected, converting.")
        ds = conv_lon_to_180(ds)

    if ds.cf["latitude"].values[0] > ds.cf["latitude"].values[1]:
        logger.info("90 to -90 latitude detected, chaning slice order.")
        lat_slice = lat_slice[::-1]
    else:
        lat_slice = lat_slice

    sel_ds = ds.cf.sel(
        latitude=slice(*lat_slice),
        longitude=slice(*lon_slice),
        time=slice(start_year, end_year),
    ).chunk("auto")

    return sel_ds


def compute_ar_pr_values(
    ar_ds: xr.Dataset,
    precip_ds: xr.Dataset,
    path: str,
    n_clusters: int,
    resample_id: str,
    client: Client,
) -> tuple[list[xr.DataArray], list[xr.DataArray], np.ndarray]:
    """Compute AR and PR values for each cluster id. Useful for plotting."""
    cluster_ds = xr.open_zarr(path).load().labels
    cluster_labels, cluster_counts = np.unique(cluster_ds, return_counts=True)

    ar_vals = []
    pr_vals = []
    for cluster_id in range(n_clusters):
        ar_vals.append(
            xr.where(
                ar_ds.resample(time=resample_id)
                .sum()
                .isel(time=cluster_ds == cluster_id)
                .ar_unique_id
                > 0,
                1,
                0,
            ).sum("time")
        )
        pr_vals.append(
            precip_ds.resample(time=resample_id)
            .sum()
            .isel(time=cluster_ds == cluster_id)
            .tp.sum("time")
        )

    ar_vals = client.compute(ar_vals)
    pr_vals = client.compute(pr_vals)

    ar_vals = [ar_val.result() for ar_val in ar_vals]
    pr_vals = [pr_val.result() for pr_val in pr_vals]

    return ar_vals, pr_vals, cluster_counts
