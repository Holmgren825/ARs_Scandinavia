import logging
from pathlib import Path

import cf_xarray
import xarray as xr
from dask.distributed import Client

PRECIP_PATH_PATTERN = (
    "/data/era5/total_precipitation/total_precipitation-1979_2020.zarr"
)
# Change this if other extent.
RESAMPEL_ZARR_PATH = (
    "/data/era5/total_precipitation/total_precipitation-1979_2020-6h.zarr"
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Resample ERA5 hourly total prectiptation to 6 hour sums."""
    logging.basicConfig(
        filename=Path(__file__).parent / "logs/resample_era5_precip.log",
        level=logging.INFO,
    )
    client = Client()
    logging.info("Opening precip dataset.")
    ds = xr.open_zarr(PRECIP_PATH_PATTERN)

    ds_resample = ds.tp.cf.resample(time="6h").sum()
    # NOTE: Can't do the rechunk at the same time, or can we?
    ds_resample = ds_resample.chunk("auto")

    logging.info("START: Saving resampled data to zarr.")
    ds_resample.to_zarr(RESAMPEL_ZARR_PATH)
    logging.info("END: Saving resampled data to zarr.")

    client.shutdown()


if __name__ == "__main__":
    main()
