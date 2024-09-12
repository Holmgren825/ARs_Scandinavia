import shutil

import xarray as xr
import zarr
from dask.diagnostics import ProgressBar
from rechunker import rechunk

INPUT_STORES = "/data/era5/total_precipitation/total_precipitation_*.zarr"
MAX_MEM = "40GB"
VAR_NAME = "tp"


def main() -> None:

    ds = xr.open_mfdataset(INPUT_STORES, parallel=True, engine="zarr")

    # Have to chunk the data since it doesn't have any chunks.
    ds_sel = ds.chunk("auto")
    ds_sel[VAR_NAME].encoding = {}

    target_chunks = {
        VAR_NAME: {"time": 61364, "latitude": 11, "longitude": 48},
        # Don't rechunk these arrays
        "time": None,
        "longitude": None,
        "latitude": None,
    }

    target_store = (
        "/data/era5/total_precipitation/total_precipitation_no-chunk-time.zarr"
    )
    temp_store = "/data/era5/total_precipitation/group_rechunked-tmp.zarr"

    # Need to remove the existing temp stores or it won't work
    shutil.rmtree(temp_store)

    array_plan = rechunk(
        ds_sel, target_chunks, MAX_MEM, target_store, temp_store=temp_store
    )

    with ProgressBar():
        array_plan.execute()

    zarr.consolidate_metadata(target_store)


if __name__ == "__main__":
    main()
