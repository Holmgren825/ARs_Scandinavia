"""Collapse common AR across time."""

import logging
import os
from pathlib import Path

import cf_xarray as cfxr
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from dask.distributed import Client
from tqdm import tqdm

ARTMIP_PATHS = [
    "/data/projects/atmo_rivers_scandinavia/ERA5.Mundhenk_v3/",
    "/data/projects/atmo_rivers_scandinavia/ERA5.Reid500/",
    "/data/projects/atmo_rivers_scandinavia/ERA5.GuanWaliser_v2/",
    "/data/projects/atmo_rivers_scandinavia/ERA5.TempestLR/",
]
STORE_PATTERN = "*.scand_ars.tracked.*.zarr"

STATS_CSV_PATH = "/data/projects/atmo_rivers_scandinavia/collapsed_stats_all.csv"

START_YEAR = "1980"
END_YEAR = "2019"
OVERWRITE = False
MAXIMUM_AR_LENGTH = 300

logger = logging.getLogger(__name__)


def main() -> None:
    """Collapse tracked AR to single samples."""
    logging.basicConfig(
        filename=Path(__file__).parent / "logs/collapse_tracked_ars.log",
        level=logging.INFO,
    )
    client = Client()

    if OVERWRITE or not os.path.exists(STATS_CSV_PATH):
        stats_df = pd.DataFrame(
            columns=["ardt_name", "ar_id", "start_date", "end_date"]
        )
    else:
        stats_df = pd.read_csv(STATS_CSV_PATH)

    for artmip_path in tqdm(ARTMIP_PATHS):
        ardt_name = os.path.basename(os.path.normpath(artmip_path))
        logger.info(f"START: {ardt_name}")
        store_path_collapsed = (
            artmip_path
            + ardt_name
            + f".scand_ars.collapsed.{START_YEAR}-{END_YEAR}.zarr"
        )

        if not os.path.exists(store_path_collapsed) or OVERWRITE:
            curr_ardt_path = artmip_path + STORE_PATTERN
            tracked_ars_da = xr.open_mfdataset(
                curr_ardt_path, engine="zarr"
            ).ar_tracked_id
            tracked_ars_da = tracked_ars_da.sel(time=slice(START_YEAR, END_YEAR))

            logger.info("START: Loading unique ids and indices.")
            unique_ids, ids_indices = da.unique(
                tracked_ars_da.fillna(0).data, return_index=True
            )
            unique_ids, ids_indices = client.compute([unique_ids, ids_indices])
            unique_ids, ids_indices = [
                res.result() for res in [unique_ids, ids_indices]
            ]

            time_idxs, _, _ = np.unravel_index(ids_indices, tracked_ars_da.shape)

            # Skip zero.
            unique_ids = unique_ids[1:]
            time_idxs = time_idxs[1:]

            logger.info("END: Loading unique ids and indices.")
            for i, (id, time_idx) in enumerate(
                tqdm(zip(unique_ids, time_idxs, strict=True), total=len(unique_ids))
            ):
                # Boolean mask
                curr_id_da = (
                    tracked_ars_da.isel(
                        time=slice(time_idx, time_idx + MAXIMUM_AR_LENGTH)
                    )
                    == id
                )
                # Select dates where we have some values.
                time_mask = curr_id_da.cf.any(["longitude", "latitude"])
                curr_id_da = curr_id_da.sel(time=time_mask)
                first_timestep_raw = curr_id_da.isel(time=[0]).time
                first_timestep = first_timestep_raw.dt.strftime("%Y%m%dT%H").values[0]
                last_timestep_raw = curr_id_da.isel(time=[-1]).time
                last_timestep = last_timestep_raw.dt.strftime("%Y%m%dT%H").values[0]

                sample_id = "-".join([f"{id}", first_timestep, last_timestep])

                # Collapse the tracked ar over time.
                collapsed_sample = curr_id_da.sum("time")
                # Assign the new sample id.
                collapsed_sample = collapsed_sample.expand_dims(sample_id=[sample_id])

                logger.debug(f"Saving collapsed sample: {sample_id}")
                if not i:
                    collapsed_sample.to_zarr(store_path_collapsed, mode="w")
                else:
                    collapsed_sample.to_zarr(
                        store_path_collapsed, append_dim="sample_id"
                    )

                new_row = pd.DataFrame(
                    {
                        "ardt_name": ardt_name,
                        "ar_id": sample_id,
                        "start_date": first_timestep_raw.values,
                        "end_date": last_timestep_raw.values,
                    },
                    index=[len(stats_df)],
                )
                stats_df = pd.concat([stats_df, new_row], ignore_index=True)
        else:
            logger.info(f"{store_path_collapsed} already exists.")

        # Make sure to save the csv after every ardt.
        stats_df.to_csv(STATS_CSV_PATH)

    client.shutdown()
    logger.info("Work finished.")


if __name__ == "__main__":
    main()
