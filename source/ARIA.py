from Image import Image
import numpy as np
import math

class ARIA(Image):

    def __init__(self, c_bounds):
        self.c_bounds = c_bounds

    """ Helper functions """

    def histogram_match(s_band, r_band, s_shape):
        s_band = np.ndarray.flatten(s_band)
        r_band = np.ndarray.flatten(r_band)

        s_value, s_idx, s_count = np.unique(s_band, return_inverse=True, return_counts=True)
        r_value, r_count = np.unique(r_band, return_counts=True)
        s_quantile = np.cumsum(s_count).astype(np.float64) / s_band.size
        r_quantile = np.cumsum(r_count).astype(np.float64) / r_band.size

        interp = np.interp(s_quantile, r_quantile, r_value)
        t_band = interp[s_idx]
        t_band = t_band.reshape((s_shape[0], s_shape[1]))

        return t_band

    """ Some functions for making different kinds of coherence difference maps """

    def simple_map(arr, t=0):
        return np.where(arr > t, arr, 0.)

    def binary_map(arr, t=0):
        flood = np.where(arr > t, 1., 0.).astype(np.int8)
        return flood

    def purple_map(arr, t=0): # assume causalty constraint
        rgba = np.zeros((arr.shape[0], arr.shape[1], 4)).astype(np.float32)
        pos = np.where(arr > 0.0)
        x = pos[0]
        y = pos[1]
        for i in range(len(x)):
            value = arr[x[i], y[i]]
            if value >= t:
                rgba[x[i], y[i], 3] = 1.0 # set completely opaque
                rgba[x[i], y[i], 0] = 0.5
                rgba[x[i], y[i], 2] = 1.0
        return rgba

    """ Driver to process using the ARIA method """

    # REQUIRED PARAMETERS
    # reference - Image object for the reference image
    # secondarys - list of Image objects for any and all secondary images
    # t - threshold used to determine whether loss of coherence is high enough to write to a map
    # map_type - choose a function to generate a numpy array that represents a colored or binary map for ARIA results

    # OPTIONAL PARAMETERS
    # map_directory - where to save any KML or GeoTIFF files
    # file_prefix - prefix to all files being written to disk

    # RETURNS
    # save_data - a list of SaveARIA objects
    def process_ARIA(self, reference, secondarys, t, map_type, map_directory=None, file_prefix=''):
        save_data= []
        r_str = reference.path.split('/')[-1] # grab file name
        r_str = r_str.split('.')[0] # remove file extension
        bounds = reference.c2b(self.c_bounds)
        for i in range(len(secondarys)):
            s_str = secondarys[i].path.split('/')[-1]
            s_str = s_str.split('.')[0]
            filename = file_prefix + '_' + r_str + '_to_' + s_str # not including file extension

            s_band, geo_bounds = self.crop(secondarys[i], bounds)
            r_band, _ = self.crop(reference, bounds)
            t_band = self.histogram_match(s_band, r_band, s_band.shape)
            diff = np.subtract(r_band, t_band)
            coh_map = map_type(diff, t)

            save_point = SaveData(coh_map, filename, geo_bounds, reference.raster)
            save_data.append(save_point)

        return save_data
