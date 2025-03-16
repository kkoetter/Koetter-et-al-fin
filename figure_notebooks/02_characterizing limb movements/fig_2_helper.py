
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colors
import matplotlib.pyplot as plt

import numpy as np


# Load the Set2 colormap
cmap = plt.get_cmap('Set2')

# Get two colors from the colormap
color1 = cmap(0)  # Get the first color
color2 = cmap(3)  # Get the second color
color3 = cmap(7)  # Get the second color

# print("Color 1:", color1)
# print("Color 2:", color2)

color_ipsi_cont = [ color1, color2, color3]


labels_cat = ['approach_swim',
 'slow1',
 'slow2',
 'burst_swim',
 'J_turn',
 'high_angle_turn',
 'routine_turn',
 'spot_avoidance_turn',
 'O_bend',
 'long_latency_C_start',
 'C_start']


color =  ['#82cfff',
  '#4589ff',
  '#0000c8',
  '#fcaf6d',
  '#ffb3b8',
  '#08bdba',
  '#24a148',
  '#9b82f3',
  '#ee5396',
  '#e3bc13',
  '#fa4d56']

color_bouts =  ['#82cfff',
  '#4589ff',
  '#0000c8',
  '#fcaf6d',
  '#ffb3b8',
  '#08bdba',
  '#24a148',
  '#9b82f3',
  '#ee5396',
  '#e3bc13',
  '#fa4d56']

cmp_bouts = colors.ListedColormap(color)



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

def create_extrema_dict(t_data, l_data, r_data):
    extrema_dict = {
        0: {
            'peaks_i': t_data[0],
            'peaks_a': t_data[1],
            'valleys_i': t_data[2],
            'valleys_a': t_data[3]
        },
        1: {
            'peaks_i': l_data[0],
            'peaks_a': l_data[1],
            'valleys_i': l_data[2],
            'valleys_a': l_data[3]
        },
        2: {
            'peaks_i': r_data[0],
            'peaks_a': r_data[1],
            'valleys_i': r_data[2],
            'valleys_a': r_data[3]
        }
    }
    return extrema_dict

def invert_peaks_val(peaks_i_array, valleys_i_array, laterality):
    # Create a copy of the traces to keep the original unchanged
    cutoff = np.full(peaks_i_array.shape[0], np.nan)

    for i, dir_value in enumerate(laterality):
        if dir_value == 0: 
            cutoff[i] = peaks_i_array[i][0]
        elif dir_value == 1: 
            cutoff[i] = valleys_i_array[i][0]
    return cutoff


# Then, for each fish_Id, you calculate the index
def calculate_index(group):
    count_0 = group.get(0, 0)  # get the count for 0, and if not present, assume it to be 0
    count_1 = group.get(1, 0)  # same for 1
    total_counts = count_0 + count_1
    if total_counts > 0:
        if count_0 == total_counts:
            return -1  # only right fin used
        elif count_1 == total_counts:
            return 1  # only left fin used
        else:
            return (count_1 - count_0) / total_counts  # both fins used
    else:
        return 0  # no fins used
        

def simplify_fish_id(fish_id):
    if '_' in fish_id:
        return '_'.join(fish_id.split('_')[:2])
    return fish_id

