import os
import datetime
import time
import glob
from subprocess import Popen, PIPE, STDOUT
import shutil

# Where the Sentinel 1 Toolbox graphing tool exe is located
baseSNAP = '/opt/snap/bin/gpt'


def timestamp(date):
    return time.mktime(date.timetuple())


# making the output directory
def output_dir(gran):
    Output_Directory = "../Main/" + 'GRD_Processed' + '/' + gran + "_Processed"
    if not os.path.exists(Output_Directory):
        os.makedirs(Output_Directory)
    return Output_Directory


# Apply precise orbit file
def applyOrbit(new_dir, granule_path_zip, granule):
    aoFlag = ' Apply-Orbit-File '
    oType = '-PcontinueOnFail=\"false\" -PorbitType=\'Sentinel Precise (Auto Download)\' '
    out = '-t ' + new_dir + '/' + granule + '_OB '
    cmd = baseSNAP + aoFlag + out + oType + granule_path_zip
    print('Applying Precise Orbit file')
    print(cmd)
    p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
    output = p.stdout.read()
    print(output)
    orbit_corrected_file_path = new_dir + '/' + granule + '_OB.dim'
    return orbit_corrected_file_path


# remove border noise
def applyremovebordernoise(new_dir, in_data_path, baseGran):
    Noise_flag = '  Remove-GRD-Border-Noise '
    out = '-t ' + new_dir + '/' + baseGran + '_OB_GBN '
    in_data_cmd = '-SsourceProduct=' + in_data_path
    cmd = baseSNAP + Noise_flag + out + in_data_cmd
    print(cmd)
    p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
    output = p.stdout.read()
    print(output)
    border_noise_file_path = new_dir + '/' + baseGran + '_OB_GBN.dim'
    return border_noise_file_path


# apply calibrations
def applyCal(new_dir, in_data_path, baseGran):
    calFlag = ' Calibration -PoutputBetaBand=false -PoutputSigmaBand=true '
    out = '-t ' + new_dir + '/' + baseGran + '_OB_GBN_CAL '
    in_data_cmd = '-Ssource=' + in_data_path
    cmd = baseSNAP + calFlag + out + in_data_cmd
    print('Applying Calibration')
    print(cmd)
    p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
    output = p.stdout.read()
    print(output)
    calibrated_file_path = new_dir + '/' + baseGran + '_OB_GBN_CAL.dim'
    return calibrated_file_path


# apply a speckle filter
#  -PfilterSizeX=7 -PfilterSizeY=7
def applySpeckle(new_dir, in_data_path, baseGran):
    speckle_flag = ' Speckle-Filter -Pfilter="Refined Lee" '
    out = '-t ' + new_dir + '/' + baseGran + '_OB_GBN_CAL_SP '
    in_data_cmd = '-Ssource=' + in_data_path
    cmd = baseSNAP + speckle_flag + out + in_data_cmd
    print(cmd)
    p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
    output = p.stdout.read()
    print(output)
    speckle_filter_file_path = new_dir + '/' + baseGran + '_OB_GBN_CAL_SP.dim'
    return speckle_filter_file_path


# Apply range doppler terrain correction
def applyTC(new_dir, in_data_path, baseGran, pixsiz, extDEM):
    tcFlag = ' Terrain-Correction '
    out = '-t ' + new_dir + '/' + baseGran + '_OB_GBN_CAL_SP_TC '
    in_data_cmd = '-Ssource=' + in_data_path + ' '
    in_data_cmd = in_data_cmd + '-PsaveDEM=false '
    in_data_cmd = in_data_cmd + '-PsaveProjectedLocalIncidenceAngle=false '
    in_data_cmd = in_data_cmd + '-PpixelSpacingInMeter=' + str(pixsiz) + ' '
    # zone, cm, hemi,
    #     if hemi == "S":
    #         in_data_cmd = in_data_cmd + '-PmapProjection=EPSG:327%02d ' % zone
    #     else:
    #         in_data_cmd = in_data_cmd + '-PmapProjection=EPSG:326%02d ' % zone

    if extDEM != " ":
        in_data_cmd = in_data_cmd + ' -PdemName=\"External DEM\" -PexternalDEMFile=%s -PexternalDEMNoDataValue=0 ' % extDEM
    else:
        in_data_cmd = in_data_cmd + ' -PdemName=\"SRTM 1Sec HGT\" '
    cmd = baseSNAP + tcFlag + out + in_data_cmd
    print('Applying Terrain Correction -- This will take some time')
    print(cmd)
    p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
    output = p.stdout.read()
    print(output)
    terrain_correction_file_path = new_dir + '/' + baseGran + '_OB_GBN_CAL_SP_TC.dim'
    return terrain_correction_file_path


# write files to tiff
def Sigma0_FF_2_gtif(new_dir, Sigma0_directory, granule):
    Sigma0_VV_path = Sigma0_directory + '/' + 'Sigma0_VV.img'
    Sigma0_VH_path = Sigma0_directory + '/' + 'Sigma0_VH.img'
    Sigma0_VV_save = new_dir + '/' + granule + '_' + Sigma0_VV_path.split('/')[-1].split('.')[0] + '.tif'
    Sigma0_VH_save = new_dir + '/' + granule + '_' + Sigma0_VH_path.split('/')[-1].split('.')[0] + '.tif'
    cmd_VV = '/home/csimmons/miniconda3/envs/sar/bin/gdal_translate -of GTiff ' + Sigma0_VV_path + ' ' + Sigma0_VV_save
    cmd_VH = '/home/csimmons/miniconda3/envs/sar/bin/gdal_translate -of GTiff ' + Sigma0_VH_path + ' ' + Sigma0_VH_save
    print(cmd_VV)
    print(cmd_VH)
    p_VV = Popen(cmd_VV, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
    p_VH = Popen(cmd_VH, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
    output_VV = p_VV.stdout.read()
    output_VH = p_VH.stdout.read()
    print(output_VV)
    print(output_VH)


def Process_GRD_File(GRD_input_list, pixsiz=10.0, extDEM_path=" "):
    """ Function that preprocesses the amplitude data
    :param GRD_input_list: Input list of GRDH data
    :param pixsiz: desired pixel size
    :param extDEM_path: If using an external DEM this is the path
    :return: pre processed geotifs in GRD_Processed folder
    """
    start_time = datetime.datetime.now()
    if pixsiz == " ":
        pixsiz = 10.0
    if extDEM_path == " ":
        print("No external DEM file specified")
    else:
        print("external DEM has been specified")
    for GRD_file in GRD_input_list:
        granule = GRD_file.split('/')[-1].split('.')[0]
        Output_Directory = output_dir(granule)
        # orbit correction
        Orbit_Correction = applyOrbit(Output_Directory, GRD_file, granule)
        # border noise removal
        Border_Noise_Removal = applyremovebordernoise(Output_Directory, Orbit_Correction, granule)
        # Calibration to sigma nought
        Calibration = applyCal(Output_Directory, Border_Noise_Removal, granule)
        # speckle filter
        Speckle_Filter = applySpeckle(Output_Directory, Calibration, granule)
        # terrain correction
        Terrain_Correction = applyTC(Output_Directory, Speckle_Filter, granule, pixsiz, extDEM_path)
        # write out data to geotiffs VV and VH
        # Sigma0_directory = dB_Conversion.replace('.dim', '.data')
        Sigma0_directory = Terrain_Correction.replace('.dim', '.data')
        Sigma0_FF_2_gtif(Output_Directory, Sigma0_directory, granule)
        # clean up
#         dir_processed = glob.glob(Output_Directory + '/' + 'S1?*.data')
#         dir_processed = [x.replace('\\', '/') for x in glob.glob(Output_Directory + '/' + 'S1?*.data')]
#         if Sigma0_directory in dir_processed:
#             dir_processed.remove(Sigma0_directory)
#         for dir_delete in dir_processed:
#             try:
#                 shutil.rmtree(dir_delete)
#             except:
#                 print('part of remove failed')
    end_time = datetime.datetime.now()
    Total_time = timestamp(end_time) - timestamp(start_time)
    return Total_time
