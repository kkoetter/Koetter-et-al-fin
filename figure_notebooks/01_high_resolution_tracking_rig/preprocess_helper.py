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