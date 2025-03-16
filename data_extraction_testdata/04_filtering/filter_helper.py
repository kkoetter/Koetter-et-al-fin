import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

def smooth_trace(trace, wnd=6, poly=2):
    return savgol_filter(trace, wnd, poly)  # window size 5, polynomial order 2


def process_data(tail, i_fins, c_fins, laterality):
    print (tail.shape)
    # Create empty lists to store processed tensors
    tails = []
    ipsilateral_fins = []
    contralateral_fins = []
    
    # Process tail data
    for i in range(len(laterality)):
        if laterality[i] == 1:
            tails.append(tail[i])
        elif laterality[i] == -1:
            tails.append(tail[i] * -1)
    
    # Process ipsilateral fins data
    for i in range(len(laterality)):
        if laterality[i] == 1:
            ipsilateral_fins.append(i_fins[i]*-1)
        elif laterality[i] == -1:
            ipsilateral_fins.append(i_fins[i])
    
    # Process contralateral fins data
    for i in range(len(laterality)):
        if laterality[i] == 1:
            contralateral_fins.append(c_fins[i])
        elif laterality[i] == -1:
            contralateral_fins.append(c_fins[i]* -1)
    
    # Convert lists to numpy arrays
    tails = np.array(tails)
    ipsilateral_fins = np.array(ipsilateral_fins)
    contralateral_fins = np.array(contralateral_fins)
    print (tails.shape)
    
    return tails, ipsilateral_fins, contralateral_fins


def process_data(tail, i_fins, c_fins, laterality):
    print(tail.shape)

    # Reshape `laterality` to be a column vector for broadcasting
    laterality = laterality.reshape(-1, 1)

    # Apply transformations based on laterality
    tails = tail * laterality
    ipsilateral_fins = i_fins * -laterality
    contralateral_fins = c_fins * laterality

    print(tails.shape)  # This should match the input shape

    return tails, ipsilateral_fins, contralateral_fins




def filter_trials(tails, ipsilateral_fins, contralateral_fins, threshold, baseline_points=5, max_threshold_factor=1.5):
    """
    Filters trials where both the start and end points have an absolute mean below a threshold,
    and removes tails whose maximum point is significantly larger than the mean maximum of all traces.
    
    Parameters:
    - tails, ipsilateral_fins, contralateral_fins: 2D numpy arrays where each row is a trace.
    - threshold: The threshold value to compare against.
    - baseline_points: Number of timepoints to consider for baseline calculation (default is 5).
    - max_threshold_factor: The factor above the mean maximum value to consider as significantly larger (default is 1.5).
    
    Returns:
    - Filtered 2D numpy arrays of tails, ipsilateral_fins, and contralateral_fins.
    - A boolean mask (`final_mask_trials`) indicating which original trials passed the filtering.
    """
    
    # Calculate the absolute mean for start and end points of the baseline
    baseline_start_tails = np.mean(np.abs(tails[:, :baseline_points]), axis=1)
    baseline_end_tails = np.mean(np.abs(tails[:, -baseline_points:]), axis=1)
    
    baseline_start_ipsi = np.mean(np.abs(ipsilateral_fins[:, :baseline_points]), axis=1)
    baseline_end_ipsi = np.mean(np.abs(ipsilateral_fins[:, -baseline_points:]), axis=1)
    
    baseline_start_contra = np.mean(np.abs(contralateral_fins[:, :baseline_points]), axis=1)
    baseline_end_contra = np.mean(np.abs(contralateral_fins[:, -baseline_points:]), axis=1)

    # Determine which trials pass the baseline threshold condition for both start and end
    valid_baseline_tails = (baseline_start_tails <= threshold) & (baseline_end_tails <= threshold)
    valid_baseline_ipsi = (baseline_start_ipsi <= threshold) & (baseline_end_ipsi <= threshold)
    valid_baseline_contra = (baseline_start_contra <= threshold) & (baseline_end_contra <= threshold)

    # Combine all baseline masks to filter the entire trial
    initial_valid_mask = valid_baseline_tails & valid_baseline_ipsi & valid_baseline_contra

    # Filter all traces by the combined initial mask
    filtered_tails = tails[initial_valid_mask]
    filtered_ipsilateral_fins = ipsilateral_fins[initial_valid_mask]
    filtered_contralateral_fins = contralateral_fins[initial_valid_mask]

    # Calculate the mean maximum of all valid tails
    if filtered_tails.size > 0:
        mean_max_tails = np.mean(np.max(filtered_tails, axis=1))
        significant_max_threshold = mean_max_tails * max_threshold_factor

        # Remove tails whose maximum point is significantly larger than the mean maximum
        valid_max_mask_tails = np.max(filtered_tails, axis=1) <= significant_max_threshold

        # Apply this mask to the initially filtered data
        final_valid_mask = initial_valid_mask.copy()
        final_valid_mask[initial_valid_mask] = valid_max_mask_tails
    else:
        # In case no trials pass the initial filtering, all are invalid
        final_valid_mask = np.full(tails.shape[0], False)

    # Apply final masks to traces
    filtered_tails = tails[final_valid_mask]
    filtered_ipsilateral_fins = ipsilateral_fins[final_valid_mask]
    filtered_contralateral_fins = contralateral_fins[final_valid_mask]

    return filtered_tails, filtered_ipsilateral_fins, filtered_contralateral_fins, final_valid_mask


