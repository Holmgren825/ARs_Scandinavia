"""Combine ERA5 total precipitation."""

import logging
import shutil
from pathlib import Path

import cf_xarray
import xarray as xr
import zarr  # type: ignore
from dask.diagnostics import ProgressBar
from distributed.client import Client
from rechunker import rechunk  # type: ignore
from tqdm.autonotebook import tqdm

PRECIP_PATH_PATTERN = "/data/era5/total_precipitation/total_precipitation_*.zarr"
# Change this if other extent.
VAR_NAME = "tp"
MAX_MEM = "5GB"
FIRST_YEAR = 1979
LAST_YEAR = 2020

TARGET_STORE = "/data/era5/total_precipitation/total_precipitation-1979_2020.zarr"
TEMP_STORE = "/data/era5/total_precipitation/group_rechunked-tmp.zarr"

logger = logging.getLogger(__name__)


def main() -> None:
    """Save ERA5 hourly total prectiptation to zarr."""
    logging.basicConfig(
        filename=Path(__file__).parent / "logs/resample_era5_precip.log",
        level=logging.INFO,
    )
    client = Client()
    logging.info("Opening precip dataset.")
    ds = xr.open_mfdataset(PRECIP_PATH_PATTERN, engine="zarr")

    for year in tqdm(range(FIRST_YEAR, LAST_YEAR + 1)):
        ds_sel = ds.sel(valid_time=f"{year}").chunk(
            "auto"
            # {"valid_time": 1594, "latitude": 66, "longitude": 262}
        )

        if year == 1979:
            ds_sel.to_zarr(TARGET_STORE)
        else:
            ds_sel.to_zarr(TARGET_STORE, mode="a", append_dim="valid_time")

    client.shutdown()


if __name__ == "__main__":
    main()
