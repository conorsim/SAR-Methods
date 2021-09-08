import numpy as np
import math
import matplotlib.pyplot as plt
import simplekml
import rasterio
from rasterio import Affine

class SaveData:
        def __init__(self, arr, filename, geo_bounds, raster):
            self.map = arr
            self.filename = filename
            self.geo_bounds = geo_bounds
            self.raster = raster

class Image(SaveData):
    """ Initialization and reading """

    def __init__(self, path):
        self.path = path
        self.raster = None
        self.band = None
        self.shape = None

    # reads the raster metadata of an image and the numpy array of a given band index
    def read(self, read_band=True, band_num=1):
        self.raster = rasterio.open(self.path)
        if read_band:
            self.band = self.raster.read(band_num)
            self.shape = self.band.shape

    """ Data preprocessing """

    # clips the data to lower and upper numeric bounds
    def threshold_clip(self, lower=None, upper=None):
        if lower:
            new_band = np.where(self.band < lower, lower, self.band)
        if upper:
            new_band = np.where(self.band > upper, upper, self.band)
        try new_band:
            self.band = new_band

    # clips the data to lower and upper quantiles
    def quantile_clip(self, lower_quantile=None, upper_quantile=None):
        if lower_quantile:
            lower_q = np.quantile(self.band, lower_quantile)
            new_band = np.where(self.band < lower_q, lower_q, self.band)
        if upper_quantile:
            upper_q = np.quantile(self.band, upper_quantile)
            new_band = np.where(self.band > upper_q, upper_q, self.band)
        try new_band:
            self.band = new_band

    # maps the numeric values in an image to a different interval defined by [a, b]
    def map_to_interval(self, a, b):
        min = np.min(self.band)
        max = np.max(self.band)
        frac = (b-a) / (max-min)
        new_band = a + frac*(self.band - min)
        self.band = new_band

    # convert data from a linear scale to a decibel scale
    def convert_to_dB(self):
        new_band = np.where(self.band != 0., 10.*np.log10(self.band), 0.)
        self.band = new_band

    """ Saving data to disk """

    def create_save(self, arr, filename, geo_bounds, raster):
        return SaveData(arr, filename, geo_bounds, raster)

    def save_kml(self, save_point, directory):
        png_path = directory + '/' + save_point.filename + '.png'
        kml_path = directory + '/' + save_point.filename + '.kml'
        plt.imsave(png_path, save_point.map)
        kml = simplekml.Kml()
        layer = kml.newgroundoverlay(name=name)
        layer.icon.href = save_point.filename
        layer.gxlatlonquad.coords = save_point.geo_bounds
        kml.save(kml_path)

    def save_tif(self, save_point, directory):
        tif_path = directory + '/' + save_point.filename + '.tif'
        x_pix_size, y_pix_size = save_point.raster.transform[0], save_point.raster.transform[4]
        left, bottom, right, top = self.c_bounds[0], self.c_bounds[2], self.c_bounds[1], self.c_bounds[3]
        cols = math.ceil(abs((right - left) / x_pix_size)) + 25
        rows = math.ceil(abs((top - bottom) / y_pix_size)) + 25
        aff = Affine(x_pix_size, 0, left, 0, y_pix_size, top)
        tif_kwargs = save_point.raster.meta.copy()
        tif_kwargs.update({
                'width': cols,
                'height': rows,
                'transform': aff,
                'nodata': 0,
                'dtype': save_point.map.dtype
            })
        new_tif = rasterio.open(tif_path, 'w', **tif_kwargs)
        new_tif.write(save_point.map, indexes=1)
        new_tif.close()

    """ Other common methods """

    # converts geographic coordinate bounds in the form [lon_min, lon_max, lat_min, lat_max]
    # to image coordinates in the form [x_min, x_max, y_min, y_max]
    def c2b(self, c_bounds):
        forward_transform = self.raster.transform
        reverse_transform = ~forward_transform
        corner1 = reverse_transform * [c_bounds[0], c_bounds[2]]
        corner2 = reverse_transform * [c_bounds[1], c_bounds[3]]
        bounds = [corner1[0], corner2[0], corner2[1], corner1[1]]
        for i in range(len(bounds)):
            if bounds[i] < 0:
                bounds[i] = 0
            elif i == 1 and bounds[i] > self.shape[1]:
                bounds[i] = self.shape[1]
            elif i == 3 and bounds[i] > self.shape[0]:
                bounds[i] = self.shape[0]
            else:
                bounds[i] = math.floor(bounds[i])
        return bounds

    def crop(self, img, bounds):
        upper_left = img.raster.transform * (bounds[0], bounds[2])
        upper_right = img.raster.transform * (bounds[1], bounds[2])
        lower_left = img.raster.transform * (bounds[0], bounds[3])
        lower_right = img.raster.transform * (bounds[1], bounds[3])
        geo_bounds = [lower_left, lower_right, upper_right, upper_left]
        return img.band[bounds[2]:bounds[3], bounds[0]:bounds[1]], geo_bounds
