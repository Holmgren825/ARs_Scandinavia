"""Preprocessing pipeline for artmip datasets."""

import logging
from pathlib import Path

import geopandas as gpd  # type: ignore
from ar_scandinavia.preproccess import ArtmipDataset, PathDict
from dask.distributed import Client
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from tqdm.autonotebook import tqdm

ARTMIP_PATHS = [
    # "/data/atmospheric_rivers/artmip/ERA5.ar.Mundhenk_v3.1hr/",
    # "/data/atmospheric_rivers/artmip/ERA5.ar.Reid500.1hr/",
    # "/data/atmospheric_rivers/artmip/ERA5.ar.GuanWaliser_v2.1hr/",
    # NOTE: This one requires some lat/lons.
    "/data/atmospheric_rivers/artmip/ERA5.ar.TempestLR.1hr/",
]

logger = logging.getLogger(__name__)


def generate_shp() -> BaseGeometry:
    # Pytest root dir is project root dir.
    gdf = gpd.read_file(
        Path(__file__).parent
        / "../etc/ne_50_admin_0_countries/ne_50m_admin_0_countries.shp"
    )
    scand_gdf = gdf[
        (gdf["ADMIN"] == "Sweden")
        + (gdf["ADMIN"] == "Norway")
        + (gdf["ADMIN"] == "Denmark")
    ]
    scand_gdf.loc[scand_gdf.index == 88, "geometry"] = (
        scand_gdf.loc[scand_gdf.index == 88, "geometry"].iloc[0].geoms[1]
    )

    scand_shape = unary_union(scand_gdf.geometry)
    return scand_shape


def main() -> None:
    logging.basicConfig(
        filename=Path(__file__).parent / "logs/preprocess.log", level=logging.INFO
    )
    client = Client()
    """Run the preprocessing pipeline for a bunch of artmip datates."""
    for artmip_path in tqdm(ARTMIP_PATHS):
        path_dict: PathDict = {
            "artmip_dir": artmip_path,
            "project_dir": "/data/projects/atmo_rivers_scandinavia/",
            "region_name": "scand_ars",
        }
        # TODO: Can probably increase n_batch_chunks. Now memeory seems to peak around 9GB.
        ar_ds = ArtmipDataset(path_dict=path_dict, time_thin=6, overwrite=True)
        logger.info(ar_ds.ardt_raw_name)

        region_shp: BaseGeometry = generate_shp()

        ar_ds.preprocess_artmip_catalog()
        ar_ds.get_unique_ar_ids()
        ar_ds.region_shp = region_shp
        ar_ds.create_region_mask()
        ar_ds.select_region_ars()

    client.shutdown()


if __name__ == "__main__":
    main()
