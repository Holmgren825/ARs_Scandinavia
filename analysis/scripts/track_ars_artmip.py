"""Track AR features across time."""

import logging
import os
import re
from pathlib import Path

import cf_xarray as cfxr
import xarray as xr
from ar_identify.feature import remap_ar_features, track_ar_features
from ar_scandinavia.preproccess import ArtmipDataset, PathDict
from dask.distributed import Client
from process_artmip import generate_shp
from tqdm import tqdm

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
OVERWRITE = False

logger = logging.getLogger(__name__)


def main() -> None:
    """Run the AR tracking on artmip datasets."""
    logging.basicConfig(
        filename=Path(__file__).parent / "logs/tracking_ar_artmip.log",
        level=logging.INFO,
    )
    client = Client()

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

        features = ar_ds.region_ar_ds.ar_unique_id

        # NOTE: This causes some slowdown, and for the tracking, I don't think it is needed.
        # features = conv_lon_to_180(features)

        timerange_str = ar_ds._get_timerange_str(features)
        fname = ar_ds.multi_subs(
            ar_ds.ardt_raw_name, {"1hr": f"{ar_ds.time_thin}hr", "ar": "ar_id"}
        )
        fname = ar_ds.generate_filename(
            [fname, ar_ds.path_dict["region_name"], "tracked", timerange_str]
        )

        store_path = os.path.join(ar_ds.ardt_proj_dir, fname) + ".zarr"

        if not os.path.exists(store_path) or OVERWRITE:
            logger.info("START: Tracking AR features")
            mappings = track_ar_features(features, use_tqdm=True)
            logger.info("END: Tracking AR features")

            logger.info("START: Applying AR feature mappings")
            tracked_features = remap_ar_features(
                features.where(features > 0, 0).data.astype(int), mappings
            )
            logger.info("END: Applying AR feature mappings")

            tracked_features_ds = xr.DataArray(
                tracked_features, coords=features.coords, name="ar_tracked_id"
            )

            logger.info("START: Saving tracked AR features")
            tracked_features_ds.to_zarr(store_path, mode="w")
            logger.info("END: Saving tracked AR features")
        else:
            logger.info(f"{store_path} already exists, not overwriting")

    client.shutdown()


if __name__ == "__main__":
    main()
