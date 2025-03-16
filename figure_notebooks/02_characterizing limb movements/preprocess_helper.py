import numpy as np
import math
import pandas as pd


def reduce_to_pi(ar):
    """Reduce angles to the -pi to pi range"""
    return np.mod(ar + np.pi, np.pi * 2) - np.pi

def nanzscore(array, axis=0):
    return (array - np.nanmean(array, axis=axis))/np.nanstd(array, axis=axis)

def midpoint(x1, y1, x2, y2):
    return ((x1 + x2)/2, (y1 + y2)/2)

def mid_head(df):
    right_eye_posterior_x =  df['right_eye_posterior'].values[:, 0].astype('float')
    right_eye_posterior_y =  df['right_eye_posterior'].values[:, 1].astype('float')
    right_eye_anterior_x =  df['right_eye_anterior'].values[:, 0].astype('float')
    right_eye_anterior_y =  df['right_eye_anterior'].values[:, 1].astype('float')

    left_eye_posterior_x =  df['left_eye_posterior'].values[:, 0].astype('float')
    left_eye_posterior_y =  df['left_eye_posterior'].values[:, 1].astype('float')
    left_eye_anterior_x =   df['left_eye_anterior'].values[:, 0].astype('float')
    left_eye_anterior_y =   df['left_eye_anterior'].values[:, 1].astype('float')

    left_mid_eye_y = (left_eye_anterior_y+left_eye_posterior_y)/2
    left_mid_eye_x = (left_eye_anterior_x+left_eye_posterior_x)/2

    right_mid_eye_x = (right_eye_anterior_x+right_eye_posterior_x)/2
    right_mid_eye_y = (right_eye_anterior_y+right_eye_posterior_y)/2

    mid_headx, mid_heady = midpoint(left_mid_eye_x,left_mid_eye_y, right_mid_eye_x, right_mid_eye_y) #xy left, xy right

    return mid_headx, mid_heady, left_mid_eye_x, left_mid_eye_y, right_mid_eye_x, right_mid_eye_y


# Function to convert radians to degrees using the built-in function
def radians_to_degrees(radians):
    return np.degrees(radians)


#Function to simplify fish_ids

def simplify_fish_id(fish_id):
    if '_' in fish_id:
        return '_'.join(fish_id.split('_')[:2])
    return fish_id

def resort_ipsi_contra_traces(left_fin, right_fin, ipsi_fin_values):
    """
    ipsi_fin = 0, left fin is ipsi, 
    ipsi_fin = 1, right fin is ipsi_fin,
    """
    ipsi_fin = np.zeros_like(left_fin)
    contra_fin = np.zeros_like(left_fin)

    for i, dir_value in enumerate(ipsi_fin_values):
        if dir_value == 0:  # Left 
            ipsi_fin[i] = left_fin[i]
            contra_fin[i] = right_fin[i]
        elif dir_value == 1:  # Right
            ipsi_fin[i] = right_fin[i]
            contra_fin[i] = left_fin[i]
    return ipsi_fin, contra_fin

def sort_ipsi_contra_arrays(array1, array2, ipsi_indicator):
    # Determine ipsilateral and contralateral
    """
    ipsi_fin = 0, left fin is ipsi, 
    ipsi_fin = 1, right fin is ipsi_fin,
    """
    ipsi_fin = np.full(array1.shape[0], np.nan)
    contra_fin = np.full(array2.shape[0], np.nan)

    for i, dir_value in enumerate(ipsi_indicator):
        if dir_value == 0:  # Left 
            ipsi_fin[i] = array1[i]
            contra_fin[i] = array2[i]
        elif dir_value == 1:  # Right
            ipsi_fin[i] = array2[i]
            contra_fin[i] = array1[i]
    return ipsi_fin, contra_fin

def get_ipsi_contra_col(df, col1, col2, ipsi_indicator_col):
    """
    Function to separate ipsilateral and contralateral fin durations.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    col1 (str): The name of the first duration column (e.g., 'l_fin_duration').
    col2 (str): The name of the second duration column (e.g., 'r_fin_duration').
    ipsi_indicator_col (str): The name of the column indicating which fin is ipsi (0 for col1, 1 for col2).

    Returns:
    np.ndarray: Array of ipsilateral fin durations.
    np.ndarray: Array of contralateral fin durations.
    """
    # Determine ipsilateral and contralateral durations
    ipsi_fin_duration = np.where(df[ipsi_indicator_col] == 0, df[col1], df[col2])
    contra_fin_duration = np.where(df[ipsi_indicator_col] == 0, df[col2], df[col1])
    
    return ipsi_fin_duration, contra_fin_duration
    

def make_all_fins_positive(traces, ipsi_fin_id):
    """
    Ensures all fins are positive by inverting values based on `ipsi_fin_id`.

    Args:
    traces: The tensor of shape (trials, signals, time points).
    ipsi_fin_id: Array where 0 indicates 'left' ipsilateral and 1 indicates 'right'.

    Returns:
    A modified tensor with all fins made positive.
    """
    # Create a copy of the traces to keep the original unchanged
    new_traces = np.copy(traces)

    # Identify indices for inversion based on ipsi_fin_id
    ipsi_indices_to_invert = np.where(ipsi_fin_id == 0)[0]
    contra_indices_to_invert = np.where(ipsi_fin_id == 1)[0]

    # Invert new_traces[:, 1, :] for ipsi condition
    new_traces[ipsi_indices_to_invert, 1, :] = -new_traces[ipsi_indices_to_invert, 1, :]

    # Invert new_traces[:, 2, :] for contra condition
    new_traces[contra_indices_to_invert, 2, :] = -new_traces[contra_indices_to_invert, 2, :]

    return new_traces
    
def invert_tail(traces, ipsi_fin_id):
    """
    Inverts `traces[:, 0, :]` based on `ipsi_fin_id`.

    Args:
    traces: The tensor of shape (trials, signals, time points).
    ipsi_fin_id: Array where 0 indicates 'left' and 1 indicates 'right'.

    Returns:
    A modified tensor with the first signal inverted where `ipsi_fin_id` is 1.
    """
    # Create a copy of the traces to keep the original unchanged
    new_traces = np.copy(traces)

    # Identify indices for inversion based on ipsi_fin_id
    indices_to_invert = np.where(ipsi_fin_id == 1)[0]

    # Invert new_traces[:, 0, :] for the specified condition
    new_traces[indices_to_invert, 0, :] = -new_traces[indices_to_invert, 0, :]

    return new_traces

def invert_fins(fin_1, fin_2, ipsi_value):
    # Create a copy of the traces to keep the original unchanged
    ipsi_fin = np.copy(fin_1)
    contra_fin = np.copy(fin_2)

    for i, dir_value in enumerate(ipsi_value):
        if dir_value == 0:  # left
            ipsi_fin[i] = fin_1[i, :]*-1
            contra_fin[i] = fin_2[i, :]
        elif dir_value == 1: #right 
            ipsi_fin[i] = fin_1[i, :]
            contra_fin[i] = fin_2[i, :]*-1
    return ipsi_fin, contra_fin

def invert_peaks_val(peaks_i_array, valleys_i_array, laterality):
    # Create a copy of the traces to keep the original unchanged
    cutoff = np.full(peaks_i_array.shape[0], np.nan)

    for i, dir_value in enumerate(laterality):
        if dir_value == 0: 
            cutoff[i] = peaks_i_array[i][0]
        elif dir_value == 1: 
            cutoff[i] = valleys_i_array[i][0]
    return cutoff
    

def assign_leading_fin(leading_fin, ipsi_fin):
    """
    Assigns leading fins as ipsilateral or contralateral based on `leading_fin` and `ipsi_fin`.

    Args:
    leading_fin: Array indicating which fin is leading (0, 1, or 2).
    ipsi_fin: Array indicating ipsilateral fin (0 or 1).

    Returns:
    Array indicating if the leading fin is ipsilateral (0) or contralateral (1).
    """
    # Initialize the result array
    leading_fin_ipsi_contra = np.zeros(len(leading_fin))

    for i in range(len(leading_fin)):
        if leading_fin[i] == ipsi_fin[i]:
            leading_fin_ipsi_contra[i] = 0  # Ipsilateral
        else:
            leading_fin_ipsi_contra[i] = 1  # Contralateral

    return leading_fin_ipsi_contra


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