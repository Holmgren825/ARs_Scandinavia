# Workflow
Scripts (analysis/scripts) are run in the following order to generate the results:

## AR
1. process_artmip.py
2. track_ars_artmip.py
3. collapse_tracked_ars.py
4. run_artmip_collapse_kmeans.py


## Precipitation data
1. rechunk_era5.py: First get the netcdf data to annual zarr stores.
2. precip_to_zarr.py: Combine annual stores to single multi-year store.
3. resample_precip.py: Downsample the hourly data to 6 hours.

