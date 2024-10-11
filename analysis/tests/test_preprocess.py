import logging

import numpy as np
import pytest
import xarray as xr
import geopandas as gpd
from distributed.client import Client
from utils.preproccess import ArtmipDataset, PathDict
from shapely.ops import unary_union

logger = logging.getLogger(__name__)


logger.info("Started testing")

# TODO: Can we come up with some better mock data? We really don't to run
# this on the whole dataset.
path_dict: PathDict = {
    "artmip_dir": "/data/atmospheric_rivers/artmip/ERA5.ar.Mundhenk_v3.1hr/",
    "project_dir": "/data/projects/atmo_rivers_scandinavia/",
}
# TODO: making this a fixture will give each a fresh copy of the ArtmipDataset.
# Hence there will be no mutation of the data. Currently we rely on this for certain checks
# e.g. populating .ar_tag_ds.
artmip_ds = ArtmipDataset(path_dict=path_dict)


@pytest.fixture(scope="module")
def dask_client() -> None:
    client = Client()
    logger.info(f"Dask client started, {client.dashboard_link}")
    yield client
    client.close()


@pytest.fixture
def test_shp() -> None:
    # Pytest root dir is project root dir.
    gdf = gpd.read_file("./etc/ne_50_admin_0_countries/ne_50m_admin_0_countries.shp")
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


def test_path_dict() -> None:
    # NOTE: Somewhat stupid since we don't really do type checks at runtime.
    assert (
        path_dict["artmip_dir"]
        == "/data/atmospheric_rivers/artmip/ERA5.ar.Mundhenk_v3.1hr/"
    )


def test_create_atmip_dataset() -> None:
    artmip_ds = ArtmipDataset(path_dict=path_dict)
    assert isinstance(artmip_ds, ArtmipDataset)


def test_extract_ardt_name() -> None:
    # ardt_name = artmip_ds._extract_ardt_name()
    assert artmip_ds.ardt_name == "ERA5.ar_tag.Mundhenk_v3.1hr"


def test_create_ar_id_fname() -> None:
    assert artmip_ds._create_ar_id_fname(1) == "ERA5.ar_id.Mundhenk_v3.1hr"
    assert artmip_ds._create_ar_id_fname(6) == "ERA5.ar_id.Mundhenk_v3.6hr"


def test_load_artmip_ds() -> None:
    ds = artmip_ds._load_artmip_ds()
    assert isinstance(ds, xr.Dataset)
    assert list(ds.coords.keys()) == ["lat", "lon", "time"]
    assert ds.chunks


def test_get_timerange_str() -> None:
    ds = artmip_ds._load_artmip_ds()
    res = artmip_ds._get_timerange_str(ds)
    # For the specific test dataset (Mundhenk)
    assert res == "19800101-20191231"


def test_preprocess_artmip_catalog() -> None:
    # This should maybe not test the actual conversion, since it is very slow.
    # But instead check that the results match.
    artmip_ds.preprocess_artmip_catalog()
    zarr_ds = artmip_ds.ar_tag_ds
    assert isinstance(zarr_ds, xr.Dataset)
    assert list(zarr_ds.coords.keys()) == ["lat", "lon", "time"]
    assert zarr_ds.time.dt.year[0].values == 1980
    assert zarr_ds.time.dt.year[-1].values == 2019
    assert list(zarr_ds.variables.keys()) == ["ar_binary_tag", "lat", "lon", "time"]


def test_get_unique_ar_ids() -> None:
    # TODO: Maybe we want to add a slow version of this, that also does some checks on the actual data?
    # With the built in caching, this is only heavy when run for the first time.
    artmip_ds.get_unique_ar_ids()
    id_ds = artmip_ds.ar_id_ds
    assert isinstance(id_ds, xr.Dataset)
    assert list(id_ds.coords.keys()) == ["lat", "lon", "time"]
    assert id_ds.time.dt.year[0].values == 1980
    assert id_ds.time.dt.year[-1].values == 2019
    assert list(id_ds.variables.keys()) == ["ar_unique_id", "lat", "lon", "time"]
    # Check that we have the correct number of years.
    assert id_ds.time.shape == artmip_ds.ar_tag_ds.thin({"time": 6}).time.shape


def test_create_region_mask(test_shp) -> None:
    artmip_ds.region_shp = test_shp
    artmip_ds.create_region_mask()
    assert isinstance(artmip_ds.region_mask_ds, xr.DataArray)
    assert artmip_ds.region_mask_ds.shape == artmip_ds.ar_id_ds.ar_unique_id.shape[1:]

logger.info("Finished testing")
