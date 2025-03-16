### Feature extraction functions
import numpy as np
import pandas as pd
import flammkuchen as fl

from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from scipy.signal import convolve
from scipy.stats import pearsonr

import pywt
from scipy.signal import cwt, ricker
from scipy import integrate

def nanzscore(array, axis=0):
    return (array - np.nanmean(array, axis=axis))/np.nanstd(array, axis=axis)

def smooth_trace(trace, wnd=6, poly=2):
    return savgol_filter(trace, wnd, poly)  # window size 5, polynomial order 2

dt_ = 50/250

def compute_vigor(tailsum, smoothing_window_size=3, std_dev_window_size=4):
    time_series= pd.Series(tailsum)
    smooth_time_series = time_series.rolling(window=smoothing_window_size).mean()
    rolling_std = smooth_time_series.rolling(window=std_dev_window_size).std()
    rolling_std.fillna(0, inplace=True) #smooth_time_series.mean()
    return rolling_std.values

def compute_bout_dur(vigor, thresh=0.05, dt=dt_):
    arr =vigor >= thresh
    first_index = np.argmax(arr)
    last_index = len(arr) - 1 - np.argmax(arr[::-1])
    duration = (last_index - first_index)/dt
    return duration, first_index, last_index

def get_vigor_stats(vigor):
    return np.max(vigor), np.median(vigor)

def compute_oscillations(data, osc_range=7, height_cutoff=0.25):
    osc = []
    for id, series in enumerate(data):
        peaks, _ = find_peaks(series, height=height_cutoff)
        num_osc = len(peaks)  # Number of oscillations
        if 1 <= num_osc <= 6:
            osc.append(num_osc)  # add tuple with id and series
        else:
            osc.append(-1)  # add tuple with id and series
    return np.asarray(osc)

def time_of_first_peak(data, height_cutoff=0.25):
    tp_peak = np.full(data.shape[0], -1) 
    time = np.linspace(0, 250, data.shape[1])
    for id, series in enumerate(data):
        peaks, _ = find_peaks(series, height=height_cutoff)
        if len(peaks) >0:
            tp_peak[id] = time[peaks[0]]
    return tp_peak

def compute_leading_fin(l_fins, r_fins, height_cutoff=0.2):
    leading_fin = np.full(l_fins.shape[0], -1) #0 = left, 1= right, 2= equal
    
    for bout in range(l_fins.shape[0]):
        l_peaks, _ = find_peaks(l_fins[bout], height=height_cutoff)
        r_peaks, _ = find_peaks(r_fins[bout], height=height_cutoff)
        if (len(l_peaks) >0) & (len(r_peaks) >0):
            if l_peaks[0] < r_peaks[0]: #left fin leads 
                leading_fin[bout] = 0
            elif l_peaks[0] > r_peaks[0]: #right fin leads 
                leading_fin[bout] = 1
            elif l_peaks[0] == r_peaks[0]: #equal fin leads
                leading_fin[bout] = 2
    return leading_fin

def compute_corr_lag(traces1, traces2):
    corr = []
    lags = []
    for bout in range(traces1.shape[0]):
        trace1 = traces1[bout]
        trace2 = traces2[bout]
        # Using scipy
        correlation_scipy, _ = pearsonr(trace1, trace2)
        corr.append(correlation_scipy)
        correlation = np.correlate(trace1, trace2, mode='full')
        # The peak of this correlation is the lag
        lag = correlation.argmax() - (len(trace2) - 1)
        lags.append(lag)
    return np.asarray(corr), np.asarray(lags)

def calulate_frequency_with_peaks(series, height_cutoff=0.25):
    time = np.linspace(0, series.shape[0], series.shape[0])
    peaks, _ = find_peaks(series, height=height_cutoff)
    # Calculate the time difference between maxima to find frequency
    time_difference = np.diff(time[peaks])
    frequency = 1 / time_difference.mean()
    return frequency



wavelets = ['db1', 'db2', 'haar', 'coif1', 'sym2'] # List of wavelets to test

def extract_wavelet_features(signal, wavelet='db1'):
    #The 'db1' wavelet (Daubechies wavelet with one vanishing moment)
    #is used by default, but you can replace this with the name of any wavelet supported by PyWavelets.
    #This function returns the approximation coefficients (cA) and detail coefficients (cD) as separate numpy arrays.
    # These coefficients can be used as features for your machine learning model
    
    # Perform wavelet decomposition
    coeffs = pywt.dwt(signal, wavelet)
    # Separate the approximation and detail coefficients
    cA, cD = coeffs
    return cA, cD
