import os
import argparse

"""
This script converts an ISCE reference SLC from Summit container processing to an amplitude product.
Please run the script in ISCE's 'merged/' directory.
"""

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--range_looks', type=int, help="Number of range looks", required=True)
parser.add_argument('-a', '--azimuth_looks', type=int, help="Number of azimuth looks", required=True)
parser.add_argument('-d', '--dem_path', type=str, help="Path to the DEM in ENVI format used in processing", required=True)
parser.add_argument('-g', '--geocode_script_path', type=str, help="Path to the ISCE's geocodeIsce.py file", required=True)
parser.add_argument('-b', '--bbox', type=int, nargs='+', help="List of lat/lon bounding box coordinated in the order S,N,W,E", required=True)
args = parser.parse_args()

# need to be in the ISCE merged/ directory
if not os.path.exists(os.getcwd()+'/master.slc.full.vrt'):
    raise OSError("master.slc.full.vrt does not exist")

# check for the existence of the DEM
if not os.path.exists(args.dem_path):
    raise OSError("DEM does not exist in the specified location")

# check for the existence of the geocoding script
if not os.path.exists(args.geocode_script_path):
    raise OSError("ISCE's geocodeIsce.py does not exists in the specified location")

# copy DEM files to current directory
os.system(f"cp {args.dem_path}* {os.getcwd()}")

# generate ISCE files for DEM if they do not already exist
if (not os.path.exists(f"{os.getcwd()+'/'+{args.dem_path.split('/')[-1]}}.vrt")) \
    or (not os.path.exists(f"{os.getcwd()+'/'+{args.dem_path.split('/')[-1]}}.xml")):
    os.system(f"gdal2isce_xml.py -i {os.getcwd()+'/'+{args.dem_path.split('/')[-1]}}")

# convert the full SLC to ENVI format
os.system("gdal_translate -of ENVI master.slc.full.vrt reference.slc.full")

# generate ISCE files for the full SLC
os.system("gdal2isce_xml.py -i reference.slc.full")

# convert the full line of sight (incident angle) file to ENVI format
os.system("gdal_translate -of ENVI los.rdr.full.vrt los.rdr.full")

# Radiometric Normalization using band math
os.system("imageMath.py -e='abs(a)*cos(b_0*PI/180.)/cos(b_1*PI/180.)' \
    --a=reference.slc.full --b=los.rdr.full -o=reference.slc.norm -t FLOAT -s BIL")

# Multi-looking
os.system(f"looks.py -i reference.slc.norm -o reference.slc.norm.ml -r {args.range_looks} -a {args.azimuth_looks}")

# Geocoding
os.system(f"{args.geocode_script_path} -f reference.slc.norm.ml -b '{args.bbox[0]} {args.bbox[1]} {args.bbox[2]} {args.bbox[3]}' \
    -d {args.dem_path.split('/')[-1]} -m ../reference/ -s ../secondary/ -r {args.range_looks} -a {args.azimuth_looks}")
