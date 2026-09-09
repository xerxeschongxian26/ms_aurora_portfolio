# WB2 HRES-T0 source notes

Store: `gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr`
Additional data: `https://huggingface.co/microsoft/aurora/blob/main/aurora-0.25-static.pickle`

- This file is found on the Weatherbench2 Cloud storage
- The dimensions of 1440 x 721 corresponds to the 0.25 degree resolution
- The format of the dimensions above are longitude x latitude
- The model, Aurora 0.25 Fine-Tuned, is the product of fine-tuning Aurora 0.25 Pretrained on this dataset
- HRES_T0 is NOT the same as HRES_Analysis
    - HRES_Analysis contain an additional surface assimilation step

## Variables & Name Mapping

The .zarr file contains 18 variables

| Aurora Name  | WB2 Name |  Units | Group (surf/atmos/static/meta) |
|---|---|---|---|
|*10u*|10m_u_component_of_wind | m s**-1 |surf|
|*10v*|10m_v_component_of_wind | m s**-1 |surf|
|---|10m_wind_speed | --- |---|
|*2t*|2m_temperature | K |surf|
|*z*|geopotential | m**2 s**-2 |atmos|
|*lat*|latitude | --- |meta|
|*lon*|longitude | --- |meta|
|*level*|level | --- |meta|
|*msl*|mean_sea_level_pressure | Pa |surf|
|*q*|specific_humidity | kg kg**-1 |atmos|
|---|surface_pressure | Pa |---|
|*t*|temperature | K |atmos|
|---|time | hours since 2016-01-01 in 6 hour increments |---|
|---|total_precipitation_6hr | m |---|
|*u*|u_component_of_wind | m s**-1 |atmos|
|*v*|v_component_of_wind | m s**-1 |atmos|
|---|vertical_velocity | Pa s**-1 |---|
|---|wind_speed | --- |---|


The .pickle file contains the following:

| Aurora Name  | WB2 Name |  Units | Group (surf/atmos/static/meta) |
|---|---|---|---|
|*lsm*| Land-Sea Mask | |static|
|*slt*| Soil Type | |static|
|*z*| Geopotential at the surface | |static|

## Coordinate conventions

| Coordinate | Dimension | Native convention | Aurora-required convention | Remap needed? |
|---|---|---|---|---|
| `latitude` (HRES-T0 zarr) | 721 | Increasing from -90 to + 90 | Strictly decreasing from +90 to -90 | Yes (surf/atmos + `lat`) |
| `latitude` (ERA5 static pickle) | 721 | Decreasing from +90 to -90 | Strictly decreasing from +90 to -90 | No |
| `longitude` | 1440 | Increasing from 0 to 359.8 | Strictly increasing from 0 to 360 (exclusive) | No |

## Pressure levels

- 13 Levels present in store: [50,  100,  150,  200,  250,  300,  400,  500,  600,  700,  850,  925, 1000]
- Aurora's required 13 levels: `50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000`

## Time resolution, In and out-sample periods & Init-point selection

- Time resolution every 6 hours
    - Time stamps have hour values in [00, 06, 12, 18]
    - Earliest available date: 2016-01-01T00:00:00.000000000
    - Latest available date: 2023-01-10T18:00:00.000000000
- In-Sample (used to fine-tune Aurora 0.25 Pretrained)
    - 2016-01-01T00:00:00.000000000
    - 2021-12-31T18:00:00.000000000
- Out-Sample (used to test inference of Aurora 0.25 Fine-Tuned)
    - 2022-01-01T00:00:00.000000000
    - 2022-12-31T18:00:00.000000000
- A 10-day roll out is performed from each initialisation point
    - i.e., The intialisation timestamp corresponds to the $t$ time stamp in ($t-6$,$t$)
    - Since the previous timestemp is always 6 hours prio, this matches the time resolution of this dataset, corresponding to previous time step
    - initialisation points correspond to all timestamps at 00UTC and 12UTC in the in-sample period

## Static variables

- Static variables such as the land-sea mask, the soil-type ad surface level geopotential stored as separate pickled datasets on the HuggingFace repository
    - Note the dimensions of the static variables have to match that of the dataset it will be joined with
    - i.e., Static variables should also have dimensions 721 x 1440
- File Details
    - HF Repo name: `microsoft/aurora`
    - File name: `aurora-0.25-static.pickle`
    - File revision: `0be7e57c685dac86b78c4a19a3ab149d13c6a3dd`

## References

- [`docs/benchmark-target.md`](benchmark-target.md)
- [`docs/decisions/0004-hres-t0-source-of-record.md`](decisions/0004-hres-t0-source-of-record.md)
- [`src/aurora_inference/contract.py`](../src/aurora_inference/contract.py)
