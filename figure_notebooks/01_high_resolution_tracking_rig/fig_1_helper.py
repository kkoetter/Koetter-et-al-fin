### Megabouts information
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colors
import math
import pandas as pd
import cv2 
from scipy.interpolate import interp1d

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

labels_cat_short = ['AS','S1','S2','BS','JT','HAT','RT','SAT','OB','LLC','SLC']

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


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


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


def reduce_to_pi(ar):
    """Reduce angles to the -pi to pi range"""
    return np.mod(ar + np.pi, np.pi * 2) - np.pi

def clean_tail(tail_x, tail_y, rolling=True,  wnd=5, thresh=100):
    new_tail_x = tail_x[:,:].copy()
    new_tail_y = tail_y[:,:].copy()

    #find regions with peaks and filter with median
    for segment in range(new_tail_x.shape[0]):
        test = np.argwhere([abs(np.diff(tail_x[segment,:],prepend=[0]))>=thresh])[:,1]
        for i in test[1:]:
            if i <= wnd:
                pass
            else:
                new_tail_x[segment, i] = np.median(tail_x[segment,i-wnd: i+wnd])

    for segment in range(new_tail_y.shape[0]):
        test = np.argwhere([abs(np.diff(tail_y[segment,:],prepend=[0]))>=thresh])[:,1]
        for i in test[1:]:
            if i <= wnd:
                pass
            else:
                new_tail_y[segment, i] = np.median(tail_y[segment,i-wnd: i+wnd])

    #add rolling window
    if rolling:
        testx = np.mean(rolling_window(new_tail_x[4,:], wnd),1)
        testy = np.mean(rolling_window(new_tail_y[4,:], wnd),1)
        new_tail_x[4,:] = np.concatenate((new_tail_x[4,:wnd-1], testx))
        new_tail_y[4,:]= np.concatenate((new_tail_y[4,:wnd-1], testy))

        testx = np.mean(rolling_window(new_tail_x[3,:], wnd),1)
        testy = np.mean(rolling_window(new_tail_y[3,:], wnd),1)
        new_tail_x[3,:] = np.concatenate((new_tail_x[3,:wnd-1], testx))
        new_tail_y[3,:]= np.concatenate((new_tail_y[3,:wnd-1], testy))
        
    return new_tail_x, new_tail_y

def exptrapolate_segments(tail_x, tail_y, N_seg):
    N_seg = 10
    T = tail_x.shape[1]
    tail_x_10 = np.zeros((N_seg+1,T))
    tail_y_10 = np.zeros((N_seg+1,T))

    for i in range(T):
        points = np.array([tail_x[:,i],tail_y[:,i]]).T  # a (nbre_points x nbre_dim) array

        alpha = np.linspace(0, 1, 11)
        # Linear length along the line:
        distance = np.cumsum( np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=1 )) )
        distance = np.insert(distance, 0, 0)/distance[-1]
        interpolator =  interp1d(distance, points, kind='cubic', axis=0)
        curve = interpolator(alpha)
        tail_x_10[:,i] = curve[:,0]
        tail_y_10[:,i] = curve[:,1]
    return tail_x_10, tail_y_10



def compute_tailsum(tail_angle):
    pre_tailsum= np.zeros((tail_angle.shape[0], tail_angle.shape[1]))
    for segment in range(tail_angle.shape[1]):
        pre_tailsum[:,segment]= (tail_angle[:, segment] - tail_angle[:,0])

    tailsum= np.sum(pre_tailsum, axis=1)/pre_tailsum.shape[1]
    tailsum = reduce_to_pi(tailsum)
    return tailsum


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


