"""Fetch AR features and centerlines."""

import dask.array as da
import numpy as np
import xarray as xr
from ar_identify.feature import _get_labels
from ar_identify.utils import uniquify_id
from dask.distributed import Client
from tqdm.autonotebook import tqdm

# TODO: Configuration should live in a separate config.toml.

# Paths
TAG_PATH = "/data/atmospheric_rivers/artmip/ERA5.ar_tag.GuanWaliser_v2.1hr.19790101-20191231.zarr"
SAVE_PATH = "/data/atmospheric_rivers/artmip/ERA5.ar_ids.GuanWaliser_v2.6hr.19790101-20191231.zarr"

# Config
FIRST_YEAR = 1979
LAST_YEAR = 2019
N_YEAR_BATCH = 1


def main() -> None:
    """"""
    client = Client()
    print(f"Client started, {client.dashboard_link}")

    get_feature_ids(client=client)


def get_feature_ids(client: Client) -> None:
    """Get AR feature ids from ARTMIP data."""

    tag_ds = xr.open_mfdataset(
        TAG_PATH, engine="zarr", chunks={"time": 1, "lat": -1, "lon": -1}
    )
    tag_ds = tag_ds.thin({"time": 6})

    # Structure
    structure = np.array(
        [np.zeros((3, 3)), np.ones((3, 3)), np.zeros((3, 3))], dtype=bool
    )

    # +1 since range upper bound is exclusive.
    for i, year in tqdm(enumerate(range(FIRST_YEAR, LAST_YEAR + 1, N_YEAR_BATCH))):

        tag_ds_sel = tag_ds.sel(time=slice(f"{year}", f"{year+N_YEAR_BATCH-1}"))

        features = _get_labels(
            tag_ds_sel.ar_binary_tag, wrap_axes=(2,), structure=structure
        )
        features = features.rechunk((1, -1, -1))

        features = da.where(
            features > 0,
            uniquify_id(tag_ds_sel["time.year"].values.reshape((-1, 1, 1)), features),
            0,
        )

        ds_feat = xr.DataArray(
            features,
            dims=("time", "lat", "lon"),
            coords={
                "time": tag_ds_sel.time,
                "lat": tag_ds_sel.lat,
                "lon": tag_ds_sel.lon,
            },
            name="ar_feautures",
        )

        print(f"Compute and save {year}...")
        if not i:
            ds_feat.to_zarr(SAVE_PATH)
        else:
            ds_feat.to_zarr(SAVE_PATH, append_dim="time")


if __name__ == "__main__":
    main()
