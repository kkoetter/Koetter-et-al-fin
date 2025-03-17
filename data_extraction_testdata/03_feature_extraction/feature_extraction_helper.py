import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu
from scipy.signal import savgol_filter
from scipy.stats import pearsonr
from skimage.filters import threshold_otsu


def extract_extrema(arr):    
    idxs = np.arange(arr.shape[0])
    min_idxs = []
    max_idxs = []
    for i in range(1, arr.shape[0] - 1):
        if arr[i - 1] < arr[i] > arr[i + 1]:
            max_idxs.append(i)
        elif arr[i - 1] > arr[i] < arr[i + 1]:
            min_idxs.append(i)
    return min_idxs, max_idxs
    
def moving_average(a, n=2):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def smooth_trace(trace, wnd=9, poly=2):
    return savgol_filter(trace, wnd, poly)  # window size 5, polynomial order 2

def find_extrema_and_peaks(data, thr=.1):
    extrema_idxs = []
    extrema_diffs = []

    for trace in data:
        # Identify extrema
        mins, maxs = extract_extrema(trace)

        # Combine, sort, and add indexes for first and last timepoints (needed to compute all needed differentials)
        extrema = np.sort(np.concatenate((np.array(mins), np.array(maxs))))
        extrema = np.insert(extrema, 0, 0)
        extrema = np.append(extrema, trace.shape[-1]-1)

        # Normalize trace to [0,1] range and compute average diff (before and after) for each extrema
        trace_norm = (trace-trace.min())/(trace.max()-trace.min())
        extrema_y = [trace_norm[i] for i in extrema]
        extrema_diff = np.abs(np.diff(extrema_y))
        avg_extrema_diff = moving_average(extrema_diff)

        # Append values
        extrema_idxs.append(extrema)
        extrema_diffs.append(avg_extrema_diff)

    # Find Otsu threshold for extrema inclusion
    # thr = threshold_otsu(np.concatenate(extrema_diffs))

    # Identify real extrema
    real_extrema = [bout > thr for bout in extrema_diffs]

    # Recapitulate positions and amplitudes for all extrema
    real_extrema_idxs = [extrema_idxs[bout][1:-1][real_extrema[bout]] for bout in range(len(real_extrema))]
    real_extrema_diff = [[np.diff(data[bout])[i] for i in real_extrema_idxs[bout]] for bout in range(len(real_extrema))]

    peaks_i = [real_extrema_idxs[bout][np.array(real_extrema_diff[bout]) < 0] for bout in range(len(real_extrema))]
    peaks_a = [[data[bout, i] for i in peaks_i[bout]] for bout in range(len(real_extrema))]

    valleys_i = [real_extrema_idxs[bout][np.array(real_extrema_diff[bout]) > 0] for bout in range(len(real_extrema))]
    valleys_a = [[data[bout, i] for i in valleys_i[bout]] for bout in range(len(real_extrema))]
    
    return peaks_a, peaks_i, valleys_a, valleys_i


def extract_peaks_valleys_arrays(data, peaks_a, peaks_i, valleys_a, valleys_i, max_n=9):
    """
    Function to extract and fill arrays for peaks and valleys.

    Parameters:
    - data: numpy.ndarray, the original data array from which peaks and valleys are derived.
    - peaks_a: list of lists, amplitudes of peaks for each trace.
    - peaks_i: list of lists, indices of peaks for each trace.
    - valleys_a: list of lists, amplitudes of valleys for each trace.
    - valleys_i: list of lists, indices of valleys for each trace.
    - max_n: int, maximum number of peaks/valleys to store for each trace (default is 9).

    Returns:
    - peaks_a_array, peaks_i_array, valleys_a_array, valleys_i_array: numpy.ndarrays filled with peak/valley data.
    """
    num_traces = data.shape[0]
    
    # Initialize arrays with NaN
    peaks_a_array = np.full((num_traces, max_n), np.nan)
    peaks_i_array = np.full((num_traces, max_n), np.nan)
    valleys_a_array = np.full((num_traces, max_n), np.nan)
    valleys_i_array = np.full((num_traces, max_n), np.nan)

    # Fill arrays with peak and valley data
    for i in range(num_traces):
        peak_count = min(len(peaks_a[i]), max_n)
        valley_count = min(len(valleys_a[i]), max_n)
        
        peaks_a_array[i, :peak_count] = peaks_a[i][:peak_count]
        peaks_i_array[i, :peak_count] = peaks_i[i][:peak_count]
        valleys_a_array[i, :valley_count] = valleys_a[i][:valley_count]
        valleys_i_array[i, :valley_count] = valleys_i[i][:valley_count]
    
    return peaks_a_array, peaks_i_array, valleys_a_array, valleys_i_array

def reshape_feature_array(feature_vector_array):
    """
    Reshape the feature vector array into a specified shape and extract sub-arrays.

    Parameters:
    - feature_vector_array: numpy.ndarray, the array to be reshaped.

    Returns:
    - reshaped_array: numpy.ndarray, the reshaped array.
    - peaks_a_array, peaks_i_array, valleys_a_array, valleys_i_array: separate sub-arrays.
    """
    max_n = int(feature_vector_array.shape[1] / 4)

    # Reshape the array
    reshaped_array = feature_vector_array.reshape(feature_vector_array.shape[0], 4, max_n)

    # Extract sub-arrays
    peaks_a_array = reshaped_array[:, 0, :]
    peaks_i_array = reshaped_array[:, 1, :]
    valleys_a_array = reshaped_array[:, 2, :]
    valleys_i_array = reshaped_array[:, 3, :]

    print(f"Reshaped array shape: {reshaped_array.shape}")

    return reshaped_array, peaks_a_array, peaks_i_array, valleys_a_array, valleys_i_array


def compute_leading_fin(l_peaks_array, r_peaks_array):
    """
    Compares the first column of two matrices row-wise and outputs an array
    with annotations:
    - 0 if the left value is smaller,
    - 1 if the right value is smaller,
    - 2 if both values are equal.
    
    Parameters:
    - l_peaks_array: 2D numpy array, where we compare the first column.
    - r_peaks_array: 2D numpy array, where we compare the first column.
    
    Returns:
    - result: 1D numpy array with annotation results for each row.
    """
    # Ensure the shapes of the arrays match and have at least one column
    if l_peaks_array.shape[0] != r_peaks_array.shape[0] or l_peaks_array.shape[1] == 0 or r_peaks_array.shape[1] == 0:
        raise ValueError("The arrays must have the same number of rows and at least one column.")

    num_rows = l_peaks_array.shape[0]
    result = np.zeros(num_rows, dtype=int)

    for i in range(num_rows):
        left_value = l_peaks_array[i, 0]
        right_value = r_peaks_array[i, 0]

        if left_value < right_value:
            result[i] = 0
        elif right_value < left_value:
            result[i] = 1
        else:
            result[i] = 2
    
    return result



def extract_durations(r_peaks_i_array, r_valleys_i_array, r_peaks_a_array, r_valleys_a_array, thr=0.2, thr_last=0.1):
    ref = np.concatenate((r_peaks_a_array, r_valleys_a_array), axis=1)
    test = np.concatenate((r_peaks_i_array, r_valleys_i_array), axis=1)
    
    starts = []
    ends = []
    durations = []
    
    for i in range(test.shape[0]):
        current_test = test[i]
        current_ref = ref[i]
        
        # Calculate start and end
        start_r = np.nanmin(current_test) if np.any(~np.isnan(current_test)) else 0
        end_r = np.nanmax(current_test) if np.any(~np.isnan(current_test)) else 0
        
        # Check the amplitude of the first point
        first_point_index = np.nanargmin(current_test) if np.any(~np.isnan(current_test)) else 0
        first_point_amplitude = current_ref[first_point_index]
        
        if abs(first_point_amplitude) > thr:
            start_r = 0
            
            # Initialize a flag for the second peak
            second_peak_found = False
            
            # Iterate through amplitudes after the first point
            for j in range(first_point_index + 1, len(current_ref)):
                amplitude_diff = abs(current_ref[j] - current_ref[j-1])
                
                if not second_peak_found:
                    # Allow finding the second peak
                    if amplitude_diff > thr_last:
                        second_peak_found = True
                else:
                    # If second peak is found, ignore small amplitude differences
                    if amplitude_diff < thr_last:
                        current_test[j] = np.nan  # Ignore this point in min/max calculation
        
            # Re-calculate end based on the modified test array
            end_r = np.nanmax(current_test) if np.any(~np.isnan(current_test)) else 0
        
        # Calculate duration
        dur_r = end_r - start_r
        
        starts.append(start_r)
        ends.append(end_r)
        durations.append(dur_r)
    
    return np.array(starts), np.array(ends), np.array(durations)

def compute_laterality(peaks_a_array, valleys_a_array):
    lateralities = np.full(peaks_a_array.shape[0],np.nan)
    # Access the first row of each array
    for bout in range(peaks_a_array.shape[0]):
        peaks_first = peaks_a_array[bout, 0]
        valleys_first = valleys_a_array[bout, 0]
        # Compare and determine laterality
        laterality = np.where( np.abs(peaks_first) > np.abs(valleys_first), 0, 1)
        lateralities[bout]= laterality
    return lateralities
    
    print("Laterality:", laterality)


def tail_oscillations_with_laterality(peaks_a_array, valleys_a_array, laterality):
    """
    Counts the total number of non-NaN elements per row in peaks_array or valleys_array
    based on the laterality for each element.
    
    Parameters:
    - peaks_a_array: 2D numpy array of peak values.
    - valleys_a_array: 2D numpy array of valley values.
    - laterality: 1D numpy array indicating laterality (0 or 1).
    
    Returns:
    - total_non_nan_count: 1D numpy array with the total count of non-NaN elements per row
      based on laterality.
    """
    total_non_nan_count = np.zeros(peaks_a_array.shape[0], dtype=float)  # Use float type

    # Count non-NaN elements based on laterality
    for i, lat in enumerate(laterality):
        if lat == 0:  # "left"
            for j in range(peaks_a_array.shape[1]):
                if not np.isnan(peaks_a_array[i, j]):
                    total_non_nan_count[i] += 1
        else:  # "right"
            for j in range(valleys_a_array.shape[1]):
                if not np.isnan(valleys_a_array[i, j]):
                    total_non_nan_count[i] += 1

    return total_non_nan_count


def combine_and_sort_indices(peaks_i_array, valleys_i_array):
    """
    Combines and sorts peaks and valleys indices for each row, ignoring NaNs.
    
    Parameters:
    - peaks_i_array: 2D numpy array of peaks indices.
    - valleys_i_array: 2D numpy array of valleys indices.
    
    Returns:
    - sorted_indices_array: 2D numpy array with sorted indices for each row.
    """
    num_rows, num_cols = peaks_i_array.shape
    sorted_indices_array = np.full((num_rows, num_cols * 2), np.nan, dtype=float)

    for row in range(num_rows):
        # Combine peaks and valleys, ignoring NaNs
        combined_indices = np.concatenate((peaks_i_array[row], valleys_i_array[row]))
        valid_indices = combined_indices[~np.isnan(combined_indices)].astype(int)
        
        # Sort the combined indices
        sorted_indices = np.sort(valid_indices)
        
        # Store the sorted indices in the output array
        sorted_indices_array[row, :len(sorted_indices)] = sorted_indices

    return sorted_indices_array

def calculate_periods_between_peaks(peaks_i_array, fps):
    """
    Calculates the periods (in seconds) between peaks for each row in the peaks_i_array, 
    maintaining the original shape of the array.
    
    Parameters:
    - peaks_i_array: 2D numpy array where each row contains indices of peaks.
    - fps: Frames per second of the sampling rate.
    
    Returns:
    - periods_array: 2D numpy array with the same shape as peaks_i_array, where each row
      contains the periods between successive peaks, and remaining positions are NaN.
    """
    num_rows, num_cols = peaks_i_array.shape
    periods_array = np.full((num_rows, num_cols), np.nan, dtype=float)

    for row in range(num_rows):
        peak_indices = peaks_i_array[row]
        # Remove NaN values and convert to integers
        valid_indices = peak_indices[~np.isnan(peak_indices)].astype(int)

        # Calculate the differences between successive peak indices
        if len(valid_indices) > 1:
            frame_differences = np.diff(valid_indices)
            # Convert frame differences to time periods in seconds
            periods = frame_differences / fps
            # Copy periods to the corresponding row in the output array
            periods_array[row, :len(periods)] = periods

    return periods_array


def compute_tbf(traces, extract_extrema, moving_average, dt=1.0, threshold=0.2, min_valid_tps=5):
    """
    Computes the TBF (time between features) for each trace based on extrema.

    Parameters:
    - traces: The 3D numpy array containing the traces to analyze.
    - extract_extrema: A function that returns the minima and maxima of a trace.
    - moving_average: A function that calculates the moving average of an array.
    - dt: Time step duration (default is 1.0).
    - threshold: The threshold for considering significant extrema changes (default is 0.2).
    - min_valid_tps: Minimum number of time points required for valid TBF computation (default is 5).

    Returns:
    - tbf_output: The computed TBF values for the input traces.
    """
    tbf_output_list = []

    for fin in range(traces.shape[1]):
        extrema_idxs = []

        for trace in traces[:, fin, :]:
            # Identify extrema
            mins, maxs = extract_extrema(trace)
            
            # Combine and sort indexes, add first and last timepoints
            extrema = np.sort(np.concatenate((np.array(mins), np.array(maxs))))
            extrema = np.insert(extrema, 0, 0)
            extrema = np.append(extrema, trace.shape[-1] - 1)

            # Normalize trace and compute average difference for each extrema
            trace_norm = (trace - trace.min()) / (trace.max() - trace.min())
            extrema_y = [trace_norm[i] for i in extrema]
            extrema_diff = np.abs(np.diff(extrema_y))
            avg_extrema_diff = moving_average(extrema_diff)

            # Append extrema if their average change is larger than the threshold
            extrema_idxs.append(extrema[1:-1][avg_extrema_diff > threshold])

        # Compute TBFs
        idxs = np.arange(traces.shape[2])
        tbf_output = np.full((traces.shape[0], idxs.shape[0]), np.nan)

        for i, extrema in enumerate(extrema_idxs):
            if len(extrema) > 1:  # To avoid issues with min() and max() on empty arrays
                valid_idxs = idxs[np.logical_and(idxs >= min(extrema), idxs < max(extrema))]

                if len(valid_idxs) > min_valid_tps:
                    time_diffs = np.diff(extrema) * dt
                    binned_tps = np.digitize(valid_idxs, extrema) - 1
                    instant_time_diff = np.array([time_diffs[i] for i in binned_tps])
                    tbf = (1 / instant_time_diff) / 2
                    tbf_output[i, valid_idxs[0]:valid_idxs[-1] + 1] = tbf

        tbf_output_list.append(tbf_output)

    return np.stack(tbf_output_list)
    
def resort_ipsi_contra(tail, left_fin, right_fin, directionality):
    """
    Resorts left_fin, right_fin, left_eye, and right_eye based on directionality
    into ipsilateral and contralateral components.
    left =0, right =1, left fin angles = positive, right fin angles= negative.
    """
    tail_new = np.zeros_like(tail)
    ipsi_fin = np.zeros_like(left_fin)
    contra_fin = np.zeros_like(left_fin)
    ipsi_fin_id = np.zeros(len(left_fin))  # Initialize with 0 (for 'left')

    for i, dir_value in enumerate(directionality):
        if dir_value == 0:  # Left direction
            tail_new[i] = tail[i]
            ipsi_fin[i] = left_fin[i]
            contra_fin[i] = right_fin[i]
            ipsi_fin_id[i] = 0  # Set as 0 (for 'left')
        elif dir_value == 1:  # Right direction
            tail_new[i] = tail[i]
            ipsi_fin[i] = right_fin[i]
            contra_fin[i] = left_fin[i]
            ipsi_fin_id[i] = 1  # Set as 1 (for 'right')
    return tail_new, ipsi_fin, contra_fin,  ipsi_fin_id



def sort_tensor_ipsi_contra(tensor, directionality):
    """
    Sort the tensor signals into ipsilateral and contralateral components
    based on the detected directionality.

    Args:
    tensor (np.ndarray): The baseline-corrected tensor of shape (trials, signals, time points).

    Returns:
    tuple: The sorted and adjusted tensor, ipsilateral fin identities, 
           and indices of trials where fins were switched.
    """
    # Extract each signal
    tail = tensor[:, 0, :]
    left_fin = tensor[:, 1, :]
    right_fin = tensor[:, 2, :]

    # Resort fins and eyes into ipsilateral and contralateral components
    tail_new, ipsi_fin, contra_fin, ipsi_fin_id = resort_ipsi_contra(
       tail, left_fin, right_fin, directionality
    )
    
    # Create a new tensor with sorted and adjusted components
    new_tensor = np.stack([tail_new, ipsi_fin, contra_fin], axis=1)
    
    return new_tensor, ipsi_fin_id

def get_eye_max(eye_rotation, eye_vergence):
    max_eye_rot = []
    max_eye_vergence = []
    
    for i in range(eye_rotation.shape[0]):
        max_eye_rot.append(np.mean(eye_rotation[i][-10:]) - np.mean(eye_rotation[i][:10]))
        max_eye_vergence.append(np.mean(eye_vergence[i][-10:]) - np.mean(eye_vergence[i][:10]))

    return max_eye_rot, max_eye_vergence


def invert_tail(data, directionality):
    to_fill = np.copy(data)
    for id, series in enumerate(data):
        if directionality[id] == 0:
            to_fill[id] = series*-1
    return to_fill


from scipy.signal import find_peaks

def compute_directionality(data, height_cutoff=0.08):
    directionaity = np.full(data.shape[0], -1) 
    for id, series in enumerate(data):
        peaks, _ = find_peaks(abs(series), height=height_cutoff)
        try:
            dir = series[peaks[np.argmax(abs(series[peaks]))]]
            if dir > 0: #> 0 , right, dir = 1
                directionaity[id] = 1
            if dir < 0: #<0, dir =0, left (invert)
                directionaity[id] = 0
        except ValueError:
            pass
    return directionaity

def compute_vigor(tailsum, smoothing_window_size=3, std_dev_window_size=4):
    time_series= pd.Series(tailsum)
    smooth_time_series = time_series.rolling(window=smoothing_window_size).mean()
    rolling_std = smooth_time_series.rolling(window=std_dev_window_size).std()
    rolling_std.fillna(0, inplace=True) #smooth_time_series.mean()
    return rolling_std.values

def compute_bout_dur(vigor, thresh=0.05, dt=0.005):
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