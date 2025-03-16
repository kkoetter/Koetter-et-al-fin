import numpy as np
import math
import pandas as pd
from scipy.interpolate import interp1d

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