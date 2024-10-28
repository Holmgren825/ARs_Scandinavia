"""KMeans pipeline for artmip datasets."""

import logging
import os
import re
from pathlib import Path

import cf_xarray as cfxr
import dask.array as da
import xarray as xr
import xeofs as xe  # type: ignore
from ar_scandinavia.preproccess import ArtmipDataset, PathDict
from ar_scandinavia.utils import subsel_ds
from dask.distributed import Client
from dask_ml.cluster import KMeans  # type: ignore
from process_artmip import generate_shp
from tqdm import tqdm
from xr_utils.coordinates import conv_lon_to_180

ARTMIP_PATHS = [
    "/data/atmospheric_rivers/artmip/ERA5.ar.Mundhenk_v3.1hr/",
    "/data/atmospheric_rivers/artmip/ERA5.ar.Reid500.1hr/",
    "/data/atmospheric_rivers/artmip/ERA5.ar.GuanWaliser_v2.1hr/",
    "/data/atmospheric_rivers/artmip/ERA5.ar.TempestLR.1hr/",
]

# We don't use the entire dataset for the spcific PCA analysis.
LAT_SLICE = (50, 74)
LON_SLICE = (-10, 45)

START_YEAR = "1980"
END_YEAR = "2019"
RESAMPLE = ["6h", "2d", "5d", "10d", "15d", "20d"]
N_CLUSTERS = 6
OVERWRITE = False

logger = logging.getLogger(__name__)


def main() -> None:
    """Run the Kmeans-clustering on artmip datasets."""
    logging.basicConfig(
        filename=Path(__file__).parent / "logs/compute_kmeans.log", level=logging.INFO
    )
    client = Client(n_workers=7, threads_per_worker=4, memory_limit="8GB")

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

        logger.info("Selecting region data.")
        ar_ds_sel = subsel_ds(
            ar_ds.region_ar_ds, LAT_SLICE, LON_SLICE, START_YEAR, END_YEAR
        )
        for resample in RESAMPLE:
            fname = get_filename(ar_ds, resample)

            path = os.path.join(ar_ds.ardt_proj_dir, fname)

            if not os.path.exists(path) or OVERWRITE:
                logger.info(f"START: kmeans {ar_ds.ardt_name}, {resample}.")
                ar_ds_sel.ar_unique_id.data = da.where(
                    ar_ds_sel.ar_unique_id.data > 0, 1, 0
                )
                ar_ds_sel = ar_ds_sel.fillna(0)

                ar_data: da.Array = (
                    ar_ds_sel.resample(time=resample).sum().ar_unique_id.data
                )
                ar_data = ar_data.reshape(ar_data.shape[0], -1)
                logger.info("Persisting AR data to cluster.")
                ar_data = ar_data.persist()

                kmeans = KMeans(n_clusters=N_CLUSTERS)
                logger.info("Fitting kmeans")
                kmeans.fit(ar_data)

                logger.info("Predicting labels.")
                labels = kmeans.predict(ar_data)

                res_ds = xr.DataArray(
                    labels, coords={"time": ar_ds_sel.time}, name="labels"
                )
                res_ds.time.attrs = {"standard_name": "time", "long_name": "Time"}

                logger.info("START: Saving labels")
                res_ds.to_zarr(path, mode="w")
                logger.info("END: Saving labels")
                logger.info("END: kmeans")
            else:
                logger.info(f"{fname} cluster labels already exists.")

    client.shutdown()


def get_filename(ar_ds: ArtmipDataset, additional_str: str | None = None) -> str:
    """Generate files for the PCA analysis results."""
    ardt_name = ar_ds.ardt_raw_name
    ardt_name = re.sub(".1hr", "", ardt_name)

    if additional_str is not None:
        filename = f"{ardt_name}.{additional_str}.cluster_labels.zarr"
    else:
        filename = f"{ardt_name}.cluster_labels.zarr"
    return filename


if __name__ == "__main__":
    main()
