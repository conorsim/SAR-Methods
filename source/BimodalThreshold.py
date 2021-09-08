from Image import Image
import numpy as np
import rasterio
from skimage import filters
from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve
import csv
import math
import matplotlib.pyplot as plt

class BimodalThreshold(Image):

    def __init__(self, c_bounds):
        self.c_bounds = c_bounds

    """ Helper functions """

    def gaussian_convolution(self, y):
        gs = [0.2261, 0.5478, 0.2261] # from Uddin paper
        pad_y = np.zeros(len(y)+2)
        pad_y[1:-1] = y
        new_y = np.zeros(len(y))
        for k in range(1,len(pad_y)-1):
            t1 = pad_y[k-1] * gs[0] # term 1
            t2 = pad_y[k] * gs[1] # term 2
            t3 = pad_y[k+1] * gs[2] # term 3
            new_y[k-1] = t1+t2+t3
        return new_y

    def count_peaks(self, x, y, cs, dcs, eps_tol=1e-16):
        peak_cnt = 0
        valley_cnt = 0
        peaks = []
        valleys = []
        for i in range(1,len(y)):
            if np.abs(y[i-1]) < eps_tol: continue
            else:
                d1 = dcs(x[i-1])
                d2 = dcs(x[i])
                sign1 = d1 / np.abs(d1)
                sign2 = d2 / np.abs(d2)
                test = sign1*sign2
                if test == -1.:
                    if sign1 == 1.:
                        peak_cnt += 1
                        peaks.append(fsolve(dcs, x[i]))
                    else:
                        valley_cnt += 1
                        valleys.append(fsolve(dcs, x[i]))
        return peak_cnt, valley_cnt, peaks, valleys

    def p1_p2_m1_m2_sb_sw_st(self, th, hist, M):
        t_vector = np.arange(0,256,1)
        # splicing
        t_lower = t_vector[0:th]
        t_higher = t_vector[th+1:-1]
        h_lower = hist[0:th]
        h_higher = hist[th+1:]
        # computations
        p1 = np.divide(np.sum(h_lower), M)
        p2 = np.divide(np.sum(h_higher), M)
        m1 = np.sum(np.divide(np.multiply(t_lower, h_lower), M*p1))
        m2 = np.sum(np.divide(np.multiply(t_higher, h_higher), M*p2))
        s12 = np.sum(np.multiply(np.square(np.subtract(t_lower, m1)), np.divide(np.divide(h_lower, M), p1)))
        s22 = np.sum(np.multiply(np.square(np.subtract(t_higher, m2)), np.divide(np.divide(h_higher, M), p2)))
        # scalar arithmetic
        sb2 = p1*p2*(m1-m2)**2
        sw2 = p1*s12 + p2*s22
        st2 = sb2 + sw2
        Bt = sb2/st2
        return Bt

    def block_arrays_list(self, Base_array, block_dim):
        row, col = Base_array.shape
        Number_row_blocks = math.ceil(row / block_dim)
        Number_col_blocks = math.ceil(col / block_dim)
        row_indexes = np.arange(0, Number_row_blocks * block_dim + 1, block_dim)
        col_indexes = np.arange(0, Number_col_blocks * block_dim + 1, block_dim)
        col_list = []
        row_list = []
        for i in range(0, len(col_indexes) - 1):
            col_s, col_e = col_indexes[i], col_indexes[i + 1]
            set_1 = str(col_s) + ':' + str(col_e)
            col_list.append(set_1)
        for j in range(0, len(row_indexes) - 1):
            row_s, row_e = row_indexes[j], row_indexes[j + 1]
            set_1 = str(row_s) + ':' + str(row_e)
            row_list.append(set_1)
        Squares_list = []
        for row_item in row_list:
            for col_item in col_list:
                item_use = row_item + ',' + col_item
                Squares_list.append(item_use)
        return Squares_list, Number_row_blocks, Number_col_blocks

    def block_arrays(self, Base_array, block_dim):
        Squares_list, Number_row_blocks, Number_col_blocks = self.block_arrays_list(Base_array, block_dim)
        N = 0
        subset_arrays = []
        for j in range(0, Number_row_blocks):
            for i in range(0, Number_col_blocks):
                square_list_str = Squares_list[N]
                square_list_row_s, square_list_row_e = int(square_list_str.split(',')[0].split(':')[0]), int(
                    square_list_str.split(',')[0].split(':')[1])
                square_list_col_s, square_list_col_e = int(square_list_str.split(',')[1].split(':')[0]), int(
                    square_list_str.split(',')[1].split(':')[1])
                subset_array = Base_array[square_list_row_s:square_list_row_e, square_list_col_s:square_list_col_e]
                subset_array[subset_array == 0] = np.nan # gets rid of NaNs
                N = N + 1
                subset_arrays.append(subset_array)
        return subset_arrays, Number_row_blocks, Number_col_blocks

    def normalize_array_and_bin(self, subset_array, N):
        subset_array[subset_array == 0] = np.nan
        subset_array_flatten = subset_array.flatten()
        subset_array_norm = (subset_array_flatten - min(subset_array_flatten))/(max(subset_array_flatten) - min(subset_array_flatten))
        bins = np.linspace(0, 1, N) # spacing of 256 discrete points between 0 and 1
        bin_count = np.histogram(subset_array_norm, bins)[0] # returns the count in each bin
        M = len(subset_array_norm)
        return bin_count, M # returns t=[0 1 2 ... 255], histogram, and number of elements in subarray

    """ Driver to process using LM and Otsu """

    def otsu_and_lm(self, images, ptf=True, v=0.1, block_dim=4000, s=500, ratio_tol=1e-2, smoothing_tol=30, verbose=False):
        otsus = []
        lms = []
        eps = np.finfo(np.float).eps # constant for machine epsilon
        for img in images:
            otsu_part = []
            lm_part = []
            print(f"Working on {img.path}")

            if ptf: band = img.band**0.1 # power transform
            else: band = img.band

            subset_arrays, Number_row_blocks1, Number_col_blocks1 = self.block_arrays(band, block_dim)
            for subset_array, i in zip(subset_arrays, range(len(subset_arrays))):
                Squares_list, Number_row_blocks, Number_col_blocks = self.block_arrays_list(subset_array, s)
                for square in Squares_list: # looping through the 'x_1:x_2,y_1:y_2' structure
                    square_list_row_s, square_list_row_e = int(square.split(',')[0].split(':')[0]), int(square.split(',')[0].split(':')[1])
                    square_list_col_s, square_list_col_e = int(square.split(',')[1].split(':')[0]), int(square.split(',')[1].split(':')[1])
                    subset = subset_array[square_list_row_s:square_list_row_e, square_list_col_s:square_list_col_e] # makes the most granular subarray or tile

                    bin_count, M = self.normalize_array_and_bin(subset, 256)
                    t_vector = np.arange(0,256,1) # vector in interval [0, 255]
                    B_vec = []
                    for th in t_vector:
                        B = self.p1_p2_m1_m2_sb_sw_st(th, bin_count, M)
                        B_vec.append(B)
                    B_vec = np.array(B_vec)
                    if len(B_vec) > 0: max_B = np.max(B_vec)
                    else: max_B = 0

                    # if the BCV condition is met
                    # added the norm condition for tiles that are mostly out of bounds
                    if max_B > 0.75 and np.linalg.norm(subset) > 100:
                        # Otsu method
                        subset_flat = subset.flatten()
                        subset_flat = subset_flat[subset_flat != 0]
                        otsu_threshold = filters.threshold_otsu(image=subset_flat, nbins=256)
                        otsu_part.append(otsu_threshold)

                        # LM method
                        y, bins = np.histogram(subset_flat, bins=256)
                        x = (bins[1:]+bins[:-1]) / 2
                        if verbose:
                            fig, ax = plt.subplots(ncols=3, figsize=(20,6))
                            ax[0].imshow(subset, cmap='gray')
                            ax[0].set_title(f'Tile of amplitude image')
                            ax[1].plot(x,y)
                            ax[1].set_title(f'Histogram with $B={max_B}$')
                        cs = CubicSpline(x, y)
                        dcs = cs.derivative()
                        peak_cnt, valley_cnt, peaks, valleys = self.count_peaks(x, y, cs, dcs)
                        count = 0
                        while peak_cnt > 2 and count < smoothing_tol:
                            count += 1
                            y = self.gaussian_convolution(y)
                            cs = CubicSpline(x, y)
                            dcs = cs.derivative()
                            peak_cnt, valley_cnt, peaks, valleys = self.count_peaks(x, y, cs, dcs)
                        lm_threshold = None
                        if peak_cnt > 1: # was if peak_cnt <= 4 and peak_cnt > 1
                            # usually extra peaks come from small, higher peaks
                            # so just looking at the first two is a good estimate for the bimodal peaks
                            peak1 = peaks[0]
                            peak2 = peaks[1]

                            for valley in valleys:
                                if valley > peak1 and valley < peak2: lm_threshold = valley
                        if lm_threshold != None:
                            lm_part.append(lm_threshold)

                        if verbose:
                            ax[2].plot(x, y)
                            ax[2].set_title(f'Otsu (red): {otsu_threshold}; LM (green): {lm_threshold}')
                            ax[2].scatter(otsu_threshold, cs(otsu_threshold), c='r')
                            ax[2].scatter(lm_threshold, cs(lm_threshold), c='g')
                            plt.show()
            otsus.append(otsu_part)
            lms.append(lm_part)
        return otsus, lms
