"""Some utilities."""

import logging

import xarray as xr
from xr_utils.coordinates import conv_lon_to_180

logger = logging.getLogger(__name__)


def subsel_ds(
    ds: xr.DataArray | xr.Dataset,
    lat_slice: tuple[int, int],
    lon_slice: tuple[int, int],
    start_year: str,
    end_year: str,
) -> xr.DataArray:
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
