"""Get Nao group frequencies for artmip datasets."""
import logging
import os
import re
from pathlib import Path
from typing import Optional

import cf_xarray as cfxr
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from ar_scandinavia.preproccess import ArtmipDataset, PathDict
from dask.distributed import Client
from numpy.typing import NDArray
from process_artmip import generate_shp
from tqdm import tqdm

ARTMIP_PATHS = [
    "/data/atmospheric_rivers/artmip/ERA5.ar.Mundhenk_v3.1hr/",
    "/data/atmospheric_rivers/artmip/ERA5.ar.Reid500.1hr/",
    "/data/atmospheric_rivers/artmip/ERA5.ar.GuanWaliser_v2.1hr/",
    "/data/atmospheric_rivers/artmip/ERA5.ar.TempestLR.1hr/",
]


OVERWRITE = False
START_YEAR = "1980"
END_YEAR = "2019"

logger = logging.getLogger(__name__)


def main() -> None:
    """Compute NAO-group frequencies for ARTMIP datasets."""
    logging.basicConfig(filename=Path(__file__).parent / "logs/compute_nao_groups.log", level=logging.INFO)


    client = Client(n_workers=6, threads_per_worker=4, memory_limit="8GB")


    for artmip_path in tqdm(ARTMIP_PATHS):
        path_dict: PathDict = {
            "artmip_dir": artmip_path,
            "project_dir": "/data/projects/atmo_rivers_scandinavia/",
            "region_name": "scand_ars",
        }
        logger.info("Initializing ARTMIP dataset.")
        ar_ds = ArtmipDataset(path_dict=path_dict, time_thin=6, overwrite=False)
        ar_ds.preprocess_artmip_catalog()
        ar_ds.get_unique_ar_ids()
        ar_ds.region_shp = generate_shp()
        ar_ds.create_region_mask()
        ar_ds.select_region_ars()

        if ar_ds.region_ar_ds is None:
            raise AttributeError("No region_ar_ds available for ArtmpDataset.")

        # INFO: Have to select the same years here.
        ar_da = ar_ds.region_ar_ds.ar_unique_id.sel(time=slice(START_YEAR, END_YEAR))
        nao_series = prepare_nao_ds(ar_da)
        nao_bins = get_nao_bins(nao_series)

        fname = get_filename(ar_ds)

        path = os.path.join(
            ar_ds.ardt_proj_dir, fname
        )
        if not os.path.exists(path) or OVERWRITE:
            logger.info("START: Computing NAO group frequencies.")

            nao_freqs = ar_da.assign_coords({"nao": ("time", nao_series.to_numpy())})
            nao_freqs = nao_freqs.groupby_bins("nao", nao_bins).map(calc_ar_freqs)
            nao_freqs = nao_freqs.compute()

            logger.info("END: Computing NAO group frequencies.")

            logger.info("START: Saving results.")
            nao_freqs["nao_bins"] = nao_freqs.nao_bins.astype(str)
            nao_freqs.to_zarr(path, mode="w")
            logger.info("END: Saving PCA results.")
        else:
            logger.info("Results already exists, skipping.")

    client.shutdown()


def get_filename(
    ar_ds: ArtmipDataset, additional_str: Optional[str] = None
) -> str:
    """Generate files for the PCA analysis results."""
    ardt_name = ar_ds.ardt_raw_name
    ardt_name = re.sub(".1hr", "", ardt_name)

    if additional_str is not None:
        filename = f"{ardt_name}.{additional_str}.nao_group_freqs.zarr"
    else:
        filename = f"{ardt_name}.nao_group_freqs.zarr"
    return filename




def calc_ar_freqs(ds: xr.DataArray) -> xr.DataArray:
    """Calculate AR frequencies for the given dataarray."""
    freqs = da.nansum(da.where(ds > 0, 1, np.nan), axis=0)
    freqs = xr.DataArray(
        freqs,
        coords={"lat": ds.lat, "lon": ds.lon},
        dims=("lat", "lon"),
        name="AR frequency",
    )
    return freqs


def prepare_nao_ds(ar_ds: xr.Dataset) -> pd.Series:
    """Prepare NAO dataframe."""
    path = Path(__file__).parent / "../etc/norm_daily_nao_index_1950_2024.txt"

    nao_df: pd.DataFrame = pd.read_csv(
        filepath_or_buffer=path,
        sep=r"\s+",
        header=None,
        names=["year", "month", "day", "nao"],
        dtype={"year": str, "month": str, "day": str, "nao": float},
        na_values="-99.0",
    )
    nao_df["time"] = pd.to_datetime(
        nao_df["year"] + "-" + nao_df["month"] + "-" + nao_df["day"]
    )
    nao_df.index = pd.Index(nao_df.time)
    nao_series = nao_df["nao"]
    nao_series = nao_series.interpolate()

    # Upsample to 6 hourly for convenience with era5 data.
    nao_series= nao_series.resample("6h").ffill()

    first_timestep = ar_ds.time[:1].to_numpy()[0]
    last_timestep = ar_ds.time[-1:].to_numpy()[0]

    # What year do we need?
    nao_series = nao_series.loc[first_timestep:last_timestep]

    return nao_series


def get_nao_bins(nao_series: pd.Series, midpoint_qtile: float = 0.4524) -> NDArray:
    """Generate NAO bins with roughly equal number of samples on either side of 0."""
    # TODO: Don't hard code midpoint.
    bins = np.quantile(nao_series, [0, midpoint_qtile / 2, midpoint_qtile, midpoint_qtile + (1 - midpoint_qtile) / 2, 1])
    return bins


if __name__ == "__main__":
    main()
