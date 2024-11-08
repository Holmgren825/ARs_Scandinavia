"""KMeans pipeline for collapsed samples of artmip datasets."""

import logging
import os
import re
from pathlib import Path

import cf_xarray as cfxr
import dask.array as da
import xarray as xr
from ar_scandinavia.utils import subsel_ds
from dask.distributed import Client
from dask_ml.cluster import KMeans  # type: ignore
from tqdm import tqdm

ARTMIP_PATHS = [
    "/data/projects/atmo_rivers_scandinavia/ERA5.Mundhenk_v3/",
    "/data/projects/atmo_rivers_scandinavia/ERA5.Reid500/",
    "/data/projects/atmo_rivers_scandinavia/ERA5.GuanWaliser_v2/",
    "/data/projects/atmo_rivers_scandinavia/ERA5.TempestLR/",
]
STORE_PATTERN = "*.scand_ars.collapsed.*.zarr"

# We don't use the entire dataset for the spcific PCA analysis.
LAT_SLICE = (50, 74)
LON_SLICE = (-10, 45)

START_YEAR = "1980"
END_YEAR = "2019"
N_CLUSTERS = 4
OVERWRITE = True

logger = logging.getLogger(__name__)


def main() -> None:
    """Run the Kmeans-clustering on collapsed artmip datasets."""
    logging.basicConfig(
        filename=Path(__file__).parent / "logs/compute_collapsed_kmeans.log",
        level=logging.INFO,
    )
    client = Client(n_workers=7, threads_per_worker=4, memory_limit="8GB")

    for artmip_path in tqdm(ARTMIP_PATHS):
        curr_ardt_path = artmip_path + STORE_PATTERN
        logger.info(f"START: {curr_ardt_path}")
        collapsed_ars_da = xr.open_mfdataset(
            curr_ardt_path, engine="zarr"
        ).ar_tracked_id

        collapsed_ars_da = subsel_ds(collapsed_ars_da, LAT_SLICE, LON_SLICE)
        collapsed_ars_da = collapsed_ars_da.fillna(0)

        ardt_name = os.path.basename(os.path.normpath(artmip_path))
        fname = get_filename(ardt_name, "collapsed")
        path = os.path.join(artmip_path, fname)

        if not os.path.exists(path) or OVERWRITE:
            logger.info(f"START: kmeans {curr_ardt_path}")

            data = collapsed_ars_da.data
            data = data.reshape(data.shape[0], -1)
            logger.info("Persisting AR to cluster.")
            data = data.persist()

            kmeans = KMeans(n_clusters=N_CLUSTERS)
            logger.info("Fitting kmeans")
            kmeans.fit(data)

            logger.info("Predicting labels.")
            labels = kmeans.predict(data)

            res_ds = xr.DataArray(
                labels, coords={"sample_id": collapsed_ars_da.sample_id}, name="labels"
            )
            res_ds["sample_id"] = res_ds.sample_id.astype("str")

            logger.info("START: Saving labels")
            res_ds.chunk("auto").to_zarr(path, mode="w")
            logger.info("END: Saving labels")
            logger.info("END: kmeans")
            # NOTE: It is good practice to realease data that has been persisted to the cluster.
            del data
        else:
            logger.info(f"{fname} cluster labels already exists.")

    client.shutdown()


def get_filename(ardt_name: str, additional_str: str | None = None) -> str:
    """Generate files for the PCA analysis results."""
    if additional_str is not None:
        filename = f"{ardt_name}.{additional_str}.cluster_labels.zarr"
    else:
        filename = f"{ardt_name}.cluster_labels.zarr"
    return filename


if __name__ == "__main__":
    main()
