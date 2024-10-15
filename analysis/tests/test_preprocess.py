import os
import logging

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from distributed.client import Client
from shapely.ops import unary_union
from utils.preproccess import ArtmipDataset, PathDict


logger = logging.getLogger(__name__)


logger.info("Started testing")


# TODO: Can we come up with some better mock data? We really don't to run
# this on the whole dataset.
path_dict: PathDict = {
    "artmip_dir": "/data/projects/atmo_rivers_scandinavia/test/ERA5.ar.Mundhenk_v3.1hr/",
    "project_dir": "/data/projects/atmo_rivers_scandinavia/test/",
    "region_name": "scand_ars",
}
# TODO: making this a fixture will give each a fresh copy of the ArtmipDataset.
# Hence there will be no mutation of the data. Currently we rely on this for certain checks
# e.g. populating .ar_tag_ds.
artmip_ds = ArtmipDataset(path_dict=path_dict, time_thin=6, overwrite=False)


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
        == "/data/projects/atmo_rivers_scandinavia/test/ERA5.ar.Mundhenk_v3.1hr/"
    )


def test_create_atmip_dataset() -> None:
    artmip_ds = ArtmipDataset(path_dict=path_dict)
    assert isinstance(artmip_ds, ArtmipDataset)


def test_extract_ardt_raw_name() -> None:
    # ardt_name = artmip_ds._extract_ardt_name()
    assert artmip_ds.ardt_raw_name == "ERA5.ar.Mundhenk_v3.1hr"


def test_extract_ardt_name() -> None:
    # ardt_name = artmip_ds._extract_ardt_name()
    assert artmip_ds.ardt_name == "ERA5.ar_tag.Mundhenk_v3.1hr"


def test_extract_ardt_dir() -> None:
    assert (
        artmip_ds.ardt_proj_dir
        == "/data/projects/atmo_rivers_scandinavia/test/ERA5.Mundhenk_v3"
    )


def test_multi_subs() -> None:
    assert (
        artmip_ds.multi_subs(artmip_ds.ardt_raw_name, {"1hr": "1hr"})
        == "ERA5.ar.Mundhenk_v3.1hr"
    )
    assert (
        artmip_ds.multi_subs(artmip_ds.ardt_raw_name, {"1hr": "6hr"})
        == "ERA5.ar.Mundhenk_v3.6hr"
    )
    assert (
        artmip_ds.multi_subs(artmip_ds.ardt_raw_name, {"1hr": "6hr", "ar": "ar_id"})
        == "ERA5.ar_id.Mundhenk_v3.6hr"
    )
    assert (
        artmip_ds.multi_subs(
            artmip_ds.ardt_raw_name, {"1hr": "3hr", "ar": "ar_id", "ERA5": "ERA6"}
        )
        == "ERA6.ar_id.Mundhenk_v3.3hr"
    )


def test_load_artmip_ds() -> None:
    ds = artmip_ds._load_artmip_ds()
    assert isinstance(ds, xr.Dataset)
    assert list(ds.coords.keys()) == ["lat", "lon", "time"]
    assert ds.chunks


def test_get_timerange_str() -> None:
    ds = artmip_ds._load_artmip_ds()
    res = artmip_ds._get_timerange_str(ds)
    # For the specific test dataset (Mundhenk)
    assert res == "19800101-19811231"


def test_generate_filename() -> None:
    fname = artmip_ds.generate_filename(["Just.a", "test", "100"])
    assert fname == "Just.a.test.100"


def test_preprocess_artmip_catalog(dask_client) -> None:
    # This should maybe not test the actual conversion, since it is very slow.
    # But instead check that the results match.
    artmip_ds.preprocess_artmip_catalog()
    zarr_ds = artmip_ds.ar_tag_ds
    assert isinstance(zarr_ds, xr.Dataset)
    assert list(zarr_ds.coords.keys()) == ["lat", "lon", "time"]
    assert zarr_ds.time.dt.year[0].values == 1980
    assert zarr_ds.time.dt.year[-1].values == 1981
    assert list(zarr_ds.variables.keys()) == ["ar_binary_tag", "lat", "lon", "time"]
    # Here we need a clever test to check for chunk uniformity.
    for dim, dim_chunksize in zarr_ds.chunksizes.items():
        logger.debug(f"dim {dim}: {dim_chunksize}")
        assert len(set(dim_chunksize)) <= 2


def test_generate_iter_blocks() -> None:
    assert_equal = np.testing.assert_equal
    test_chunks = {"time": (10, 10, 10, 5), "lat": (721,), "lon": (1440,)}
    iter_blocks = artmip_ds._generate_iter_blocks(test_chunks, 2)
    assert_equal(iter_blocks["time"], ((0, 10), (10, 20), (20, 30), (30, 35)))
    assert_equal(iter_blocks["lat"], ((0, 721), (0, 721), (0, 721), (0, 721)))
    assert_equal(iter_blocks["lon"], ((0, 1440), (0, 1440), (0, 1440), (0, 1440)))

    test_chunks = {"time": (10, 10, 10), "lat": (721,), "lon": (1440,)}
    iter_blocks = artmip_ds._generate_iter_blocks(test_chunks, 2)
    assert_equal(iter_blocks["time"], ((0, 10), (10, 20), (20, 30)))
    assert_equal(iter_blocks["lat"], ((0, 721), (0, 721), (0, 721)))
    assert_equal(iter_blocks["lon"], ((0, 1440), (0, 1440), (0, 1440)))

    test_chunks = {"time": (10, 10, 10, 10, 10, 5), "lat": (721,), "lon": (1440,)}
    iter_blocks = artmip_ds._generate_iter_blocks(test_chunks, 3)
    assert_equal(iter_blocks["time"], ((0, 20), (20, 40), (40, 55)))
    assert_equal(iter_blocks["lat"], ((0, 721), (0, 721), (0, 721)))
    assert_equal(iter_blocks["lon"], ((0, 1440), (0, 1440), (0, 1440)))

    iter_blocks = artmip_ds._generate_iter_blocks(test_chunks, 4)
    assert_equal(iter_blocks["time"], ((0, 30), (30, 55)))
    assert_equal(iter_blocks["lat"], ((0, 721), (0, 721)))
    assert_equal(iter_blocks["lon"], ((0, 1440), (0, 1440)))

    test_chunks = {
        "time": (10, 10, 10, 10, 10, 10, 10, 10, 5),
        "lat": (721,),
        "lon": (1440,),
    }
    iter_blocks = artmip_ds._generate_iter_blocks(test_chunks, 5)
    assert_equal(iter_blocks["time"], ((0, 40), (40, 80), (80, 85)))
    assert_equal(iter_blocks["lat"], ((0, 721), (0, 721), (0, 721)))
    assert_equal(iter_blocks["lon"], ((0, 1440), (0, 1440), (0, 1440)))

    test_chunks = {
        "time": (12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 5),
        "lat": (721,),
        "lon": (1440,),
    }
    iter_blocks = artmip_ds._generate_iter_blocks(test_chunks, 5)
    assert_equal(iter_blocks["time"], ((0, 48), (48, 96), (96, 125)))
    assert_equal(iter_blocks["lat"], ((0, 721), (0, 721), (0, 721)))
    assert_equal(iter_blocks["lon"], ((0, 1440), (0, 1440), (0, 1440)))

    iter_blocks = artmip_ds._generate_iter_blocks(test_chunks, 4)
    assert_equal(iter_blocks["time"], ((0, 36), (36, 72), (72, 108), (108, 125)))
    assert_equal(iter_blocks["lat"], ((0, 721), (0, 721), (0, 721), (0, 721)))
    assert_equal(iter_blocks["lon"], ((0, 1440), (0, 1440), (0, 1440), (0, 1440)))

    iter_blocks = artmip_ds._generate_iter_blocks(test_chunks, 10)
    assert_equal(iter_blocks["time"], ((0, 108), (108, 125)))
    assert_equal(iter_blocks["lat"], ((0, 721), (0, 721)))
    assert_equal(iter_blocks["lon"], ((0, 1440), (0, 1440)))

    test_chunks = {
        "time": (12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 5),
        "lat": (721,),
        "lon": (1440,),
    }
    iter_blocks = artmip_ds._generate_iter_blocks(test_chunks, 12)
    assert_equal(iter_blocks["time"], ((0, 132), (132, 185)))
    assert_equal(
        iter_blocks["lat"],
        (
            (0, 721),
            (0, 721),
        ),
    )
    assert_equal(
        iter_blocks["lon"],
        (
            (0, 1440),
            (0, 1440),
        ),
    )


def test_get_unique_ar_ids(dask_client) -> None:
    # TODO: Maybe we want to add a slow version of this, that also does some checks on the actual data?
    # With the built in caching, this is only heavy when run for the first time.
    artmip_ds.get_unique_ar_ids()
    id_ds = artmip_ds.ar_id_ds
    assert isinstance(id_ds, xr.Dataset)
    assert list(id_ds.coords.keys()) == ["lat", "lon", "time"]
    assert id_ds.time.dt.year[0].values == 1980
    assert id_ds.time.dt.year[-1].values == 1981
    assert list(id_ds.variables.keys()) == ["ar_unique_id", "lat", "lon", "time"]
    # Check that we have the correct number of years.
    assert id_ds.time.shape == artmip_ds.ar_tag_ds.time.shape


def test_create_region_mask(test_shp) -> None:
    artmip_ds.region_shp = test_shp
    artmip_ds.create_region_mask()
    assert isinstance(artmip_ds.region_mask_ds, xr.DataArray)
    assert artmip_ds.region_mask_ds.shape == artmip_ds.ar_id_ds.ar_unique_id.shape[1:]


def test_select_region_ars(dask_client) -> None:
    artmip_ds.select_region_ars()
    test_ds = artmip_ds.region_ar_ds
    assert isinstance(test_ds, xr.Dataset)
    assert test_ds.time.dt.year[0].values == 1980
    assert test_ds.time.dt.year[-1].values == 1981
    assert list(test_ds.variables.keys()) == ["ar_unique_id", "lat", "lon", "time"]
    assert artmip_ds.region_mask_ds.shape == test_ds.ar_unique_id.shape[1:]


logger.info("Finished testing")
