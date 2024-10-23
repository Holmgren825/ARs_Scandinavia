"""PCA pipeline for artmip datasets."""

import logging
import os
import re
from typing import Dict, Optional

import cf_xarray as cfxr
import dask.array as da
import xarray as xr
import xeofs as xe  # type: ignore
from ar_scandinavia.pca_utils import ComputePCA
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
ERA5_PRECIP_PATH = "/data/era5/total_precipitation/total_precipitation.zarr/"

# We don't use the entire dataset for the spcific PCA analysis.
LAT_SLICE = (50, 74)
LON_SLICE = (-10, 45)

START_YEAR = "1980"
END_YEAR = "2019"
RESAMPLE = "5d"
OVERWRITE = False

logger = logging.getLogger(__name__)


def main() -> None:
    """Run the PCA  on artmip datasets pipeline."""
    logging.basicConfig(filename="compute_pca.log", level=logging.INFO)
    client = Client(n_workers=7, threads_per_worker=4, memory_limit="8GB")

    precip_ds = xr.open_zarr(ERA5_PRECIP_PATH, decode_cf=True)

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
        ar_ds_sel = subsel_ds(ar_ds.region_ar_ds)
        pr_ds_sel = subsel_ds(precip_ds)

        if not ar_ds_sel.ar_unique_id.shape == pr_ds_sel.tp.shape:
            raise ValueError(
                f"AR and precipation data not of the same shape: {ar_ds_sel.ar_unique_id.shape},{pr_ds_sel.tp.shape}"
            )

        ar_ds_sel.ar_unique_id.data = da.where(ar_ds_sel.ar_unique_id.data > 0, 1, 0)

        ar_ds_sel = ar_ds_sel.fillna(0)
        pr_ds_sel = pr_ds_sel.fillna(0)

        logger.info("START: Persisting data before PCA computation.")
        ar_ds_sel = ar_ds_sel.persist()
        pr_ds_sel = pr_ds_sel.persist()
        logger.info("END: Persisting data before PCA computation.")

        ar_ds_sel_resample = ar_ds_sel.resample(time=RESAMPLE, label="right").sum()
        pr_ds_sel_resample = pr_ds_sel.resample(time=RESAMPLE, label="right").sum()

        pca_data = [
            ar_ds_sel.ar_unique_id,
            [ar_ds_sel.ar_unique_id, pr_ds_sel.tp],
            [ar_ds_sel_resample.ar_unique_id, pr_ds_sel_resample.tp],
        ]
        filename_args = (
            "std",
            "ERA5_precip.std",
            f"ERA5_precip.{RESAMPLE}_resample.std",
        )

        if not len(pca_data) == len(filename_args):
            raise ValueError("pca_data should be same length as filename_args")

        for data, filename_arg in zip(pca_data, filename_args, strict=False):
            model = xe.single.EOF(n_modes=10, use_coslat=True, standardize=True)

            combined_pca = isinstance(data, list)

            fnames = get_pca_filenames(ar_ds, filename_arg)

            pca_model = ComputePCA(
                data=data,
                model=model,
                base_path=ar_ds.ardt_proj_dir,
                result_fnames=fnames,
                combined_pca=combined_pca,
                normalize=False,
            )

            path = os.path.join(
                pca_model.base_path, pca_model.result_fnames["comp_name"]
            )
            if not os.path.exists(path) or OVERWRITE:
                logger.info("START: Computing PCA.")
                pca_model.fit()
                logger.info("END: Computing PCA.")
                logger.info("START: Saving PCA results.")
                pca_model.save(mode="w")
                logger.info("END: Saving PCA results.")
            else:
                logger.info("PCA results already exists, skipping.")

    client.shutdown()


def get_pca_filenames(
    ar_ds: ArtmipDataset, additional_str: Optional[str] = None
) -> Dict[str, str]:
    """Generate files for the PCA analysis results."""
    ardt_name = ar_ds.ardt_raw_name
    ardt_name = re.sub(".1hr", "", ardt_name)
    if additional_str is not None:
        filenames = {
            "comp_name": f"{ardt_name}.{additional_str}.eofs_components.zarr",
            "exp_var_name": f"{ardt_name}.{additional_str}.eofs_exp_var.zarr",
            "scores_name": f"{ardt_name}.{additional_str}.eofs_scores.zarr",
        }
    else:
        filenames = {
            "comp_name": f"{ardt_name}.eofs_components.zarr",
            "exp_var_name": f"{ardt_name}.eofs_exp_var.zarr",
            "scores_name": f"{ardt_name}.eofs_scores.zarr",
        }
    return filenames


def subsel_ds(ds: xr.DataArray) -> xr.DataArray:
    """Generate a subselection of given dataset."""
    if ds.cf["longitude"].values[-1] > 359:
        logger.info("0-360 coordinates detected, converting.")
        ds = ds.assign_coords(
            {
                ds.cf.coordinates["longitude"][0]: ((ds.cf["longitude"] - 180) % 360)
                - 180
            }
        )
        ds = ds.sortby(ds.cf["longitude"])

    if ds.cf["latitude"].values[0] > ds.cf["latitude"].values[1]:
        logger.info("90 to -90 latitude detected, chaning slice order.")
        lat_slice = LAT_SLICE[::-1]
    else:
        lat_slice = LAT_SLICE

    sel_ds = ds.cf.sel(
        latitude=slice(*lat_slice),
        longitude=slice(*LON_SLICE),
        time=slice(START_YEAR, END_YEAR),
    ).chunk("auto")

    return sel_ds


if __name__ == "__main__":
    main()
