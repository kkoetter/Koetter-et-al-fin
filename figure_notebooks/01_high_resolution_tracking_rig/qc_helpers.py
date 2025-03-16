import cv2 
import math
import numpy as np


def calculate_percentage(part, whole):
    percentage = 100 * float(part)/float(whole)
    return percentage

def calculate_percentage_lists(parts, wholes):
    if len(parts) != len(wholes):
        raise ValueError("Both lists must have the same length.")
    
    percentages = []
    for part, whole in zip(parts, wholes):
        if whole == 0:
            percentages.append(0)  # Avoid division by zero; you might want to handle this differently
        else:
            percentage = 100 * float(part) / float(whole)
            percentages.append(percentage)
    
    return percentages

def smooth_bool_array(bool_array, window_size=5):
    # Create a uniform window for convolution
    window = np.ones(window_size)/window_size
    # Convert bool array to int type and apply convolution
    smoothed = np.convolve(bool_array.astype(int), window, mode='same')
    # Convert back to boolean by applying a threshold at 0.5
    thresholded = smoothed > 0.5
    return thresholded

def quantify_periods(arr):
    return [(key, len(list(group))) for key, group in itertools.groupby(arr)]

def median_length_periods(arr):
    quantified = quantify_periods(arr)
    true_lengths = [length for key, length in quantified if key]
    false_lengths = [length for key, length in quantified if not key]
    return np.median(true_lengths), np.median(false_lengths)

def calculate_percentage(part, whole):
    percentage = 100 * float(part)/float(whole)
    return percentage

def calculate_percentage_lists(parts, wholes):
    if len(parts) != len(wholes):
        raise ValueError("Both lists must have the same length.")
    
    percentages = []
    for part, whole in zip(parts, wholes):
        if whole == 0:
            percentages.append(0)  # Avoid division by zero; you might want to handle this differently
        else:
            percentage = 100 * float(part) / float(whole)
            percentages.append(percentage)
    
    return percentages

def smooth_bool_array(bool_array, window_size=5):
    # Create a uniform window for convolution
    window = np.ones(window_size)/window_size
    # Convert bool array to int type and apply convolution
    smoothed = np.convolve(bool_array.astype(int), window, mode='same')
    # Convert back to boolean by applying a threshold at 0.5
    thresholded = smoothed > 0.5
    return thresholded

def quantify_periods(arr):
    return [(key, len(list(group))) for key, group in itertools.groupby(arr)]

def median_length_periods(arr):
    quantified = quantify_periods(arr)
    true_lengths = [length for key, length in quantified if key]
    false_lengths = [length for key, length in quantified if not key]
    return np.median(true_lengths), np.median(false_lengths)


def calculate_ratio_per_fish(clusters_, fish_ids, features):
    # Initialize a dictionary to store the ratio per fish
    ratio_per_fish = {}
    
    # Define target clusters
    target_clusters = [ 9, 10] #7, 8,
    
    # Iterate through each fish ID
    for fish_id in fish_ids:
        # Filter clusters specific to the current fish
        clusters_fish = clusters_[features.fish_id==fish_id]
        
        # Count occurrences of the target clusters
        target_count = clusters_fish.isin(target_clusters).sum()
        
        # Count total occurrences (all bouts)
        total_count = clusters_fish.count()
        
        # Calculate the ratio
        if total_count > 0:
            ratio = target_count / total_count
        else:
            ratio = 0  # Handle case where total_count is 0
        
        # Store the ratio in the dictionary
        ratio_per_fish[fish_id] = ratio
    
    return ratio_per_fish

def classify_fish(hists, fish_ids):
    mid = len(hists[0]) // 2  # assuming all arrays have the same length
    calm_fish_ids = []
    stressed_fish_ids = []
    
    for i, hist in enumerate(hists):
        first_half_sum = np.sum(hist[:mid])
        second_half_sum = np.sum(hist[mid:])
        
        if first_half_sum > second_half_sum:
            stressed_fish_ids.append(fish_ids[i])
        else:
            calm_fish_ids.append(fish_ids[i])
    
    return calm_fish_ids, stressed_fish_ids



