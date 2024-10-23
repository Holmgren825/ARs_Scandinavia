import xarray as xr
from distributed.client import Client

DATA_PATH = "/data/atmospheric_rivers/artmip/ERA5.ar.tempestLR.1hr/*.nc"


def main() -> None:
    client = Client()

    ds = xr.open_mfdataset(DATA_PATH)
    # TODO: It is possible to just add the coordinate values from a reference dataset e.g. GuanWaliser.
    # Assuming that the data haven't been changed from its original ERA5-form.

    client.shutdown()


if __name__ == "__main__":
    main()
