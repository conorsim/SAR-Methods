### Scripts for SAR Processing

#### `preprocess_snap.py`

Usage:

`python3 preprocess_snap.py -gpt [path to GPT] -data [path to data] -dem [path to DEM]`

Performs the following steps:
- Orbit Correction
- Border Noise Removal
- Radiometric Calibration
- Speckle Filter
- Terrain Correction

Notes:
- Operates on Sentinel-1 **GRDH** data
- Requires SNAP (specifically the `gpt` command line tool) and GDAL to be installed
- Essentially `gpt` wrapped in Python

#### `postprocess_isce.py`

Usage:

`python3 postprocess_isce.py -r [range looks] -a [azimuth looks] -d [path to DEM] -g [path to geocodeIsce.py ISCE script] -b [bounding box]`

Performs the following steps:
- Radiometric Normalization
- Multilooking
- Geocoding

Notes:
- Operates on ISCE-processed Sentinel-1 **SLC** data
- Requires ISCE and GDAL to be installed
- Tested in conjunction with container processing on Summit
