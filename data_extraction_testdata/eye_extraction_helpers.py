## eye extraction helper functions
# Image
# import cv2 

#eye detetcion
from skimage.segmentation import flood, flood_fill
from skimage.measure import label, regionprops, regionprops_table
import h5py
import numpy as np
import math
# my own utils
from utils_motato import compute_angle_between_vect_tail,compute_angle_between_vect,exptrapolate_segments

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

def mid_head_parallel(df):
    right_eye_posterior_x =  df['right_eye_posterior'].values[0].astype('float')
    right_eye_posterior_y =  df['right_eye_posterior'].values[1].astype('float')
    right_eye_anterior_x =  df['right_eye_anterior'].values[0].astype('float')
    right_eye_anterior_y =  df['right_eye_anterior'].values[1].astype('float')

    left_eye_posterior_x =  df['left_eye_posterior'].values[0].astype('float')
    left_eye_posterior_y =  df['left_eye_posterior'].values[1].astype('float')
    left_eye_anterior_x =   df['left_eye_anterior'].values[0].astype('float')
    left_eye_anterior_y =   df['left_eye_anterior'].values[1].astype('float')

    left_mid_eye_y = (left_eye_anterior_y+left_eye_posterior_y)/2
    left_mid_eye_x = (left_eye_anterior_x+left_eye_posterior_x)/2

    right_mid_eye_x = (right_eye_anterior_x+right_eye_posterior_x)/2
    right_mid_eye_y = (right_eye_anterior_y+right_eye_posterior_y)/2

    mid_headx, mid_heady = midpoint(left_mid_eye_x,left_mid_eye_y, right_mid_eye_x, right_mid_eye_y) #xy left, xy right

    return mid_headx, mid_heady, left_mid_eye_x, left_mid_eye_y, right_mid_eye_x, right_mid_eye_y


def preprocess_img(imgage, mid_eye_y, mid_eye_x, tol_val=10):
    #TODO optional: gaussian smoothing and crop for speed up
    img = imgage[:,:,0]
    thresh = flood_fill(img, (int(mid_eye_y),int(mid_eye_x)),1 , tolerance=10)
    mask =np.zeros((img.shape[0], img.shape[1]))
    mask[thresh ==1 ]=1
    return mask

def angles(i, mask_right, mask_left, body_x, body_y):
    #Todo, figure out why this worked better than a generalised functinon
    #RIGHT EYE
    label_img = label(mask_right)
    regions = regionprops(label_img)
    for props in regions:
        r_y0, r_x0 = props.centroid
        orientation = props.orientation
        r_x1 = r_x0 + math.cos(orientation) * 0.5 * props.minor_axis_length #orthogonal
        r_y1 = r_y0 - math.sin(orientation) * 0.5 * props.minor_axis_length #orthogonal

        r_x2 = r_x0 - math.sin(orientation) * 0.5 * props.major_axis_length #endpoint -
        r_y2 = r_y0 - math.cos(orientation) * 0.5 * props.major_axis_length #endpoint -
        r_x3 = r_x0 + math.sin(orientation) * 0.5 * props.major_axis_length #endpoint +
        r_y3 = r_y0 + math.cos(orientation) * 0.5 * props.major_axis_length #endpoint +

    r_eye_points = [(r_x2, r_y2), (r_x3, r_y3)] #endpoints of eye vector end points (point1xy, point2xy)
    r_orth_eye_points =[(r_x0, r_y0), (r_x1, r_y1)] # #endpoints of eye vector (orthogonal). This vector is just for 

    distance = math.sqrt( ((body_x[i]-r_eye_points[0][0])**2)+((body_y[i]- r_eye_points[0][1])**2) )
    distance2 = math.sqrt( ((body_x[i]-r_eye_points[1][0])**2)+((body_y[i]- r_eye_points[1][1])**2) )
    if distance > distance2: #flip points
        right_eye_anterior_x =   r_eye_points[0][0]
        right_eye_anterior_y =   r_eye_points[0][1]
    else:
        right_eye_anterior_x =   r_eye_points[1][0]
        right_eye_anterior_y =   r_eye_points[1][1]

    a = r_x0-right_eye_anterior_x 
    b = r_y0-right_eye_anterior_y
    right_eye_vect = np.array([-b, a])

    #LEFT EYE
    label_img = label(mask_left)
    regions = regionprops(label_img)
    for props in regions:
        l_y0, l_x0 = props.centroid
        orientation = props.orientation
        l_x1 = l_x0 + math.cos(orientation) * 0.5 * props.minor_axis_length #orthogonal
        l_y1 = l_y0 - math.sin(orientation) * 0.5 * props.minor_axis_length #orthogonal

        l_x2 = l_x0 - math.sin(orientation) * 0.5 * props.major_axis_length #endpoint -
        l_y2 = l_y0 - math.cos(orientation) * 0.5 * props.major_axis_length #endpoint -
        l_x3 = l_x0 + math.sin(orientation) * 0.5 * props.major_axis_length #endpoint +
        l_y3 = l_y0 + math.cos(orientation) * 0.5 * props.major_axis_length #endpoint +

    l_eye_points = [(l_x2, l_y2), (l_x3, l_y3)] #endpoints of eye vector end points (point1xy, point2xy)
    l_orth_eye_points =[(l_x0, l_y0), (l_x1, l_y1)] # #endpoints of eye vector (orthogonal). This vector is just for 

    distance = math.sqrt( ((body_x[i]-l_eye_points[0][0])**2)+((body_y[i]- l_eye_points[0][1])**2) )
    distance2 = math.sqrt( ((body_x[i]-l_eye_points[1][0])**2)+((body_y[i]- l_eye_points[1][1])**2) )
    if distance > distance2: #flip points
        left_eye_anterior_x =   l_eye_points[0][0]
        left_eye_anterior_y =   l_eye_points[0][1]
    else:
        left_eye_anterior_x =   l_eye_points[1][0]
        left_eye_anterior_y =   l_eye_points[1][1]


    a = l_x0-left_eye_anterior_x 
    b = l_y0-left_eye_anterior_y
    left_eye_vect = np.array([b, -a])
    
    return left_eye_vect, right_eye_vect, l_eye_points, r_eye_points, [l_x0, l_y0], [r_x0, r_y0]


def angles_parallel(mask_right, mask_left, body_x, body_y):
    #Todo, figure out why this worked better than a generalised functinon
    #RIGHT EYE
    label_img = label(mask_right)
    regions = regionprops(label_img)
    for props in regions:
        r_y0, r_x0 = props.centroid
        orientation = props.orientation
        r_x1 = r_x0 + math.cos(orientation) * 0.5 * props.minor_axis_length #orthogonal
        r_y1 = r_y0 - math.sin(orientation) * 0.5 * props.minor_axis_length #orthogonal

        r_x2 = r_x0 - math.sin(orientation) * 0.5 * props.major_axis_length #endpoint -
        r_y2 = r_y0 - math.cos(orientation) * 0.5 * props.major_axis_length #endpoint -
        r_x3 = r_x0 + math.sin(orientation) * 0.5 * props.major_axis_length #endpoint +
        r_y3 = r_y0 + math.cos(orientation) * 0.5 * props.major_axis_length #endpoint +

    r_eye_points = [(r_x2, r_y2), (r_x3, r_y3)] #endpoints of eye vector end points (point1xy, point2xy)
    r_orth_eye_points =[(r_x0, r_y0), (r_x1, r_y1)] # #endpoints of eye vector (orthogonal). This vector is just for 

    distance = math.sqrt( ((body_x-r_eye_points[0][0])**2)+((body_y- r_eye_points[0][1])**2) )
    distance2 = math.sqrt( ((body_x-r_eye_points[1][0])**2)+((body_y- r_eye_points[1][1])**2) )
    if distance > distance2: #flip points
        right_eye_anterior_x =   r_eye_points[0][0]
        right_eye_anterior_y =   r_eye_points[0][1]
    else:
        right_eye_anterior_x =   r_eye_points[1][0]
        right_eye_anterior_y =   r_eye_points[1][1]

    a = r_x0-right_eye_anterior_x 
    b = r_y0-right_eye_anterior_y
    right_eye_vect = np.array([-b, a])

    #LEFT EYE
    label_img = label(mask_left)
    regions = regionprops(label_img)
    for props in regions:
        l_y0, l_x0 = props.centroid
        orientation = props.orientation
        l_x1 = l_x0 + math.cos(orientation) * 0.5 * props.minor_axis_length #orthogonal
        l_y1 = l_y0 - math.sin(orientation) * 0.5 * props.minor_axis_length #orthogonal

        l_x2 = l_x0 - math.sin(orientation) * 0.5 * props.major_axis_length #endpoint -
        l_y2 = l_y0 - math.cos(orientation) * 0.5 * props.major_axis_length #endpoint -
        l_x3 = l_x0 + math.sin(orientation) * 0.5 * props.major_axis_length #endpoint +
        l_y3 = l_y0 + math.cos(orientation) * 0.5 * props.major_axis_length #endpoint +

    l_eye_points = [(l_x2, l_y2), (l_x3, l_y3)] #endpoints of eye vector end points (point1xy, point2xy)
    l_orth_eye_points =[(l_x0, l_y0), (l_x1, l_y1)] # #endpoints of eye vector (orthogonal). This vector is just for 

    distance = math.sqrt( ((body_x-l_eye_points[0][0])**2)+((body_y- l_eye_points[0][1])**2) )
    distance2 = math.sqrt( ((body_x-l_eye_points[1][0])**2)+((body_y- l_eye_points[1][1])**2) )
    if distance > distance2: #flip points
        left_eye_anterior_x =   l_eye_points[0][0]
        left_eye_anterior_y =   l_eye_points[0][1]
    else:
        left_eye_anterior_x =   l_eye_points[1][0]
        left_eye_anterior_y =   l_eye_points[1][1]


    a = l_x0-left_eye_anterior_x 
    b = l_y0-left_eye_anterior_y
    left_eye_vect = np.array([b, -a])
    
    return left_eye_vect, right_eye_vect, l_eye_points, r_eye_points, [l_x0, l_y0], [r_x0, r_y0]




def nanzscore(array, axis=0):
    return (array - np.nanmean(array, axis=axis))/np.nanstd(array, axis=axis)

def preprocess_img(imgage, mid_eye_y, mid_eye_x, tol_val=10):
    #TODO optional: gaussian smoothing and crop for speed up
    img = imgage[:,:,0]
    thresh = flood_fill(img, (int(mid_eye_y),int(mid_eye_x)),1 , tolerance=10)
    mask =np.zeros((img.shape[0], img.shape[1]))
    mask[thresh ==1 ]=1
    return mask

def compute_angles(i, left_eye_vect, right_eye_vect, body_vect):
    ## Compute angles between vectors
    left_eye_angle =  compute_angle_between_vect(body_vect[:,i], left_eye_vect)
    right_eye_angle =  compute_angle_between_vect(right_eye_vect, body_vect[:,i])
    vergence = compute_angle_between_vect(right_eye_vect,left_eye_vect)
    rotation_eye = right_eye_angle-left_eye_angle
    return left_eye_angle, right_eye_angle, rotation_eye, vergence

def compute_angles_parallel(left_eye_vect, right_eye_vect, body_vect):
    ## Compute angles between vectors
    left_eye_angle =  compute_angle_between_vect(body_vect, left_eye_vect)
    right_eye_angle =  compute_angle_between_vect(right_eye_vect, body_vect)
    vergence = compute_angle_between_vect(right_eye_vect,left_eye_vect)
    rotation_eye = right_eye_angle-left_eye_angle
    return left_eye_angle, right_eye_angle, rotation_eye, vergence