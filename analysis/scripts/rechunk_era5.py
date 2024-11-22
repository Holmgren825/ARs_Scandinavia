import shutil

import xarray as xr
import zarr  # type: ignore
from distributed.client import Client
from rechunker import rechunk  # type: ignore
from tqdm import tqdm

FIRST_YEAR = 1979
LAST_YEAR = 2020
ERA5_DATA_PATH = "/data/era5/total_precipitation/total_precipitation-*.nc"
MAX_MEM = "5GB"
VAR_NAME = "tp"

TARGET_STORE = "/data/era5/total_precipitation/total_precipitation_"
TEMP_STORE = "/data/era5/total_precipitation/group_rechunked-tmp.zarr"


def main() -> None:
    """Peform the initial zarr conversion."""
    client = Client()
    ds = xr.open_mfdataset(ERA5_DATA_PATH, parallel=True)

    for year in tqdm(range(FIRST_YEAR, LAST_YEAR + 1)):
        # Have to chunk the data since it doesn't have any chunks.
        ds_sel = ds.sel(valid_time=f"{year}").chunk("auto")

        target_chunks = {
            VAR_NAME: {"valid_time": 1594, "latitude": 66, "longitude": 262},
            # Don't rechunk these arrays
            "valid_time": None,
            "longitude": None,
            "latitude": None,
        }
        year_string = f"{year}.zarr"

        # Need to remove the existing temp stores or it won't work
        try:
            shutil.rmtree(TEMP_STORE)
        except FileNotFoundError:
            pass

        array_plan = rechunk(
            ds_sel,
            target_chunks,
            MAX_MEM,
            TARGET_STORE + year_string,
            temp_store=TEMP_STORE,
        )

        array_plan.execute()

        zarr.consolidate_metadata(TARGET_STORE + year_string)

    client.shutdown()


if __name__ == "__main__":
    main()
