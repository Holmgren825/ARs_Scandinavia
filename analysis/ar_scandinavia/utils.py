"""Some utilities."""

import logging
from typing import overload

import numpy as np
import xarray as xr
from distributed.client import Client
from xarray.core.types import Dims
from xr_utils.coordinates import conv_lon_to_180

logger = logging.getLogger(__name__)


@overload
def subsel_ds(
    ds: xr.DataArray,
    lat_slice: tuple[int, int],
    lon_slice: tuple[int, int],
    start_year: str | None = None,
    end_year: str | None = None,
) -> xr.DataArray: ...


@overload
def subsel_ds(
    ds: xr.Dataset,
    lat_slice: tuple[int, int],
    lon_slice: tuple[int, int],
    start_year: str | None = None,
    end_year: str | None = None,
) -> xr.Dataset: ...


def subsel_ds(
    ds: xr.DataArray | xr.Dataset,
    lat_slice: tuple[int, int],
    lon_slice: tuple[int, int],
    start_year: str | None = None,
    end_year: str | None = None,
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
    )
    if start_year is not None and end_year is not None:
        sel_ds = sel_ds.cf.sel(time=slice(start_year, end_year))

    # Don't remember why I added a chunk("auto") here.
    # Removed now since it messes with the chunk dependent functions.
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


def compute_ar_pr_values_collapsed(
    ar_ds: xr.Dataset,
    precip_ds: xr.Dataset,
    cluster_labels: xr.DataArray,
    n_clusters: int,
    client: Client,
) -> tuple[list[xr.DataArray], list[xr.DataArray], np.ndarray]:
    """Compute AR and PR values for each cluster id. Useful for plotting."""
    cluster_counts = []

    ar_vals = []
    pr_vals = []
    pr_time_dim_name = precip_ds.cf["time"].name
    for cluster_id in range(n_clusters):
        curr_cluster_mask = (cluster_labels == cluster_id).compute()
        times = np.asarray(
            list(
                map(
                    lambda x: x[-23:].split("-"),
                    cluster_labels[curr_cluster_mask].sample_id.values,
                )
            )
        )
        cluster_precip = []
        ar_lengths = []
        for time in times:
            precip_ar_sum = precip_ds.tp.cf.sel(
                time=slice(time[0], time[1]),
            )
            cluster_precip.append(precip_ar_sum)
            ar_lengths.append(precip_ar_sum.shape[0])

        ar_vals.append(
            (
                (ar_ds.isel(sample_id=curr_cluster_mask).ar_tracked_id > 0)
                * np.asarray(ar_lengths).reshape(-1, 1, 1)
            ).sum("sample_id")
        )

        cluster_precip_da = xr.concat(cluster_precip, dim=pr_time_dim_name)
        # Get the number of timesteps in each cluster.
        cluster_counts.append(cluster_precip_da.shape[0])
        pr_vals.append(cluster_precip_da.sum(pr_time_dim_name))

    ar_vals = client.compute(ar_vals)
    pr_vals = client.compute(pr_vals)

    ar_vals = [ar_val.result() for ar_val in ar_vals]
    pr_vals = [pr_val.result() for pr_val in pr_vals]

    return ar_vals, pr_vals, np.asarray(cluster_counts)


def compute_spatial_correltion(
    da_a: xr.DataArray, da_b: xr.DataArray, dim: Dims = None
) -> xr.DataArray:
    """Compute the weighted spatial correlation between two patterns."""
    # NOTE: Normalize the data.
    da_a = (da_a - da_a.mean()) / da_a.std()
    da_b = (da_b - da_b.mean()) / da_b.std()

    # Create weights based on the latitude.
    weights = np.cos(np.deg2rad(da_a.cf["latitude"]))

    # NOTE: Make sure that we are using the same coordinate names.
    return xr.corr(da_a, da_b.cf.rename_like(da_a), dim=dim, weights=weights)
