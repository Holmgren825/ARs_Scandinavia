"""Generate a bunch of plots for ARTMIP/precipiation collapsed clustering."""

import logging
import os
from itertools import product
from pathlib import Path

import cf_xarray as cfxr
import xarray as xr
from ar_scandinavia.plots import cluster_plot
from ar_scandinavia.utils import compute_ar_pr_values_collapsed, subsel_ds
from dask.distributed import Client
from tqdm import tqdm

PRECIP_PATH = "/data/era5/total_precipitation/total_precipitation_*.zarr/"

ARDT_NAMES = ["Mundhenk_v3", "Reid500", "GuanWaliser_v2", "TempestLR"]
BASE_PATH = "/data/projects/atmo_rivers_scandinavia/"

# We don't use the entire dataset for the spcific PCA analysis.
LAT_SLICE = (50, 74)
LON_SLICE = (-10, 45)

START_YEAR = "1980"
END_YEAR = "2019"
N_CLUSTERS = 4
OVERWRITE = True

logger = logging.getLogger(__name__)


def main() -> None:
    """Run the Kmeans-clustering on artmip datasets."""
    logging.basicConfig(
        filename=Path(__file__).parent / "logs/gen_collapsed_cluster_plots.log",
        level=logging.INFO,
    )
    client = Client(n_workers=7, threads_per_worker=4, memory_limit="8GB")
    logger.info("Opening precipitation dataset.")
    precip_ds = xr.open_mfdataset(PRECIP_PATH, engine="zarr")
    precip_ds = subsel_ds(precip_ds, LAT_SLICE, LON_SLICE, START_YEAR, END_YEAR)

    tot_pr = precip_ds.sum("time").load()

    for ardt_name in tqdm(ARDT_NAMES):
        label_path = os.path.join(
            BASE_PATH,
            f"ERA5.{ardt_name}/ERA5.{ardt_name}.collapsed.cluster_labels.zarr/",
        )
        ardt_path = (
            f"ERA5.{ardt_name}/ERA5.{ardt_name}.scand_ars.collapsed.1980-2019.zarr/"
        )

        ar_path = os.path.join(
            BASE_PATH,
            ardt_path,
        )

        fig_path = f"../../figures/ar_pr_clusters_{ardt_name}-collapsed.svg"
        if not os.path.exists(fig_path) or OVERWRITE:
            logger.info(f"Open and compute {ardt_name}")
            ar_ds = xr.open_zarr(ar_path)
            ar_ds = subsel_ds(ar_ds, LAT_SLICE, LON_SLICE)
            label_da = xr.open_zarr(label_path).labels
            ar_vals, pr_vals, cluster_counts = compute_ar_pr_values_collapsed(
                ar_ds, precip_ds, label_da, N_CLUSTERS, client
            )
            fig = cluster_plot(
                ar_vals,
                pr_vals,
                ar_ds,
                ardt_name,
                tot_pr,
                cluster_counts,
                N_CLUSTERS,
                LAT_SLICE,
                LON_SLICE,
            )
            fig.savefig(fig_path)
    client.shutdown()


if __name__ == "__main__":
    main()
