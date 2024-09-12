import shutil

import xarray as xr
import zarr
from rechunker import rechunk
from tqdm import tqdm

FIRST_YEAR = 2020
LAST_YEAR = 2020
ERA5_DATA_PATH = "/data/era5/total_precipitation/total_precipitation-*.nc"
MAX_MEM = "40GB"
VAR_NAME = "tp"


def main() -> None:
    ds = xr.open_mfdataset(ERA5_DATA_PATH, parallel=True)

    for year in tqdm(range(FIRST_YEAR, LAST_YEAR + 1)):

        # Have to chunk the data since it doesn't have any chunks.
        ds_sel = ds.sel(time=f"{year}").chunk("auto")

        target_chunks = {
            VAR_NAME: {"time": 409, "latitude": 101, "longitude": 404},
            # Don't rechunk these arrays
            "time": None,
            "longitude": None,
            "latitude": None,
        }

        target_store = f"/data/era5/total_precipitation/total_precipitation_{year}.zarr"
        temp_store = "/data/era5/total_precipitation/group_rechunked-tmp.zarr"

        # Need to remove the existing temp stores or it won't work
        shutil.rmtree(temp_store)

        array_plan = rechunk(
            ds_sel, target_chunks, MAX_MEM, target_store, temp_store=temp_store
        )

        array_plan.execute()

        zarr.consolidate_metadata(target_store)


if __name__ == "__main__":
    main()
