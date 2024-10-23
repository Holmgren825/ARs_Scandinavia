import re
from glob import glob
from typing import Any, Dict

import pandas as pd
import xarray as xr
from distributed.client import Client
from tqdm import tqdm

SHIELDS_PATH = "/data/atmospheric_rivers/artmip/ERA5.ar.Shields_v1.1hr/*.nc"


def main() -> None:
    _ = Client()

    files = glob(SHIELDS_PATH)
    for file in tqdm(files[1:]):
        print(file)
        first_date, last_date = re.findall(r"\d{8}", file)
        # Have to add the hour for the end.
        last_date += " 23:00"
        new_time = pd.date_range(first_date, last_date, freq="h")
        attrs = {"long_name": "time", "standard_name": "time"}
        ds = xr.open_mfdataset(file, decode_times=False, chunks="auto")
        ds = ds.assign_coords(time=new_time)
        ds.time.attrs = attrs
        new_filename = re.sub(".nc", ".tfix.nc", file)
        print(new_filename)
        # encoding: Dict[str, Any] = {"zlib": True, "complevel": 5}
        ds.to_netcdf(new_filename)


if __name__ == "__main__":
    main()
