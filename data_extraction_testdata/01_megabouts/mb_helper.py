import numpy as np


def calculate_angles(body_vectors, midhead_vectors):
    """
    Calculate the angles between pairs of vectors: body_vectors and midhead_vectors.

    Parameters:
    - body_vectors: numpy.ndarray of shape (n, 2), each row representing (body_x, body_y).
    - midhead_vectors: numpy.ndarray of shape (n, 2), each row representing (midhead_x, midhead_y).

    Returns:
    - angles: numpy.ndarray of shape (n,), the angles in radians between the pairs of vectors.
    """
    # Calculate the dot products for each pair
    dot_products = np.sum(body_vectors * midhead_vectors, axis=1)
    
    # Calculate magnitudes of the vectors
    body_magnitudes = np.linalg.norm(body_vectors, axis=1)
    midhead_magnitudes = np.linalg.norm(midhead_vectors, axis=1)
    
    # Calculate angles in radians
    cosine_angles = dot_products / (body_magnitudes * midhead_magnitudes)
    # Handle potential numerical stability issues
    cosine_angles = np.clip(cosine_angles, -1.0, 1.0)
    
    angles_radians = np.arccos(cosine_angles)
    
    return angles_radians

def compute_angle_between_vect_tail(v1, v2):
    dot = np.einsum('ijk,ijk->ij',[v1,v1,v2],[v2,v1,v2])
    cos_= dot[0,:]
    sin_= np.cross(v1,v2)
    angle_= np.arctan2(sin_,cos_)
    return angle_


def compute_angle_between_vect(u,v):
    u = u/np.linalg.norm(u)
    v = v/np.linalg.norm(v)
    return np.arctan2(u[0]*v[1]-u[1]*v[0],u[0]*v[0]+u[1]*v[1])


def exptrapolate_segments(tail_x, tail_y, N_seg):
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

def midpoint(x1, y1, x2, y2):
    return ((x1 + x2)/2, (y1 + y2)/2)

def mid_head(right_eye_posterior_x, right_eye_posterior_y,
             right_eye_anterior_x, right_eye_anterior_y,
             left_eye_posterior_x, left_eye_posterior_y,
             left_eye_anterior_x, left_eye_anterior_y
):
    left_mid_eye_y = (left_eye_anterior_y+left_eye_posterior_y)/2
    left_mid_eye_x = (left_eye_anterior_x+left_eye_posterior_x)/2

    right_mid_eye_x = (right_eye_anterior_x+right_eye_posterior_x)/2
    right_mid_eye_y = (right_eye_anterior_y+right_eye_posterior_y)/2

    mid_headx, mid_heady = midpoint(left_mid_eye_x,left_mid_eye_y, right_mid_eye_x, right_mid_eye_y) #xy left, xy right

    return mid_headx, mid_heady, left_mid_eye_x, left_mid_eye_y, right_mid_eye_x, right_mid_eye_y

def tail_angles(tail_x_10, tail_y_10, body_x, body_y, N_seg):
    # Tail angle computations
    vect_segment = np.concatenate((np.diff(tail_x_10,axis=0)[:,:,np.newaxis],np.diff(tail_y_10,axis=0)[:,:,np.newaxis]),axis=2)
    vect_segment = np.swapaxes(vect_segment,0,2)

    start_vect = np.vstack((tail_x_10[0,:]-body_x,tail_y_10[0,:]-body_y)).T
    body_vect = -np.vstack((tail_x_10[0,:]-body_x,tail_y_10[0,:]-body_y)).T
    body_angle = np.arctan2(body_vect[:,1],body_vect[:,0])
    body_angle = np.unwrap(body_angle)

    relative_angle = np.zeros((vect_segment.shape[1],N_seg))
    start_vect = np.vstack((tail_x_10[0,:]-body_x,tail_y_10[0,:]-body_y)).T

    for i in range(N_seg):
        relative_angle[:,i] = compute_angle_between_vect_tail(start_vect,vect_segment[:,:,i].T)#,vect_segment[:,:,i+1].T)
        start_vect = np.copy(vect_segment[:,:,i].T)

    tail_angle=np.cumsum(relative_angle,1)

    return tail_angle, body_angle


def fin_preprocess(df, body_angle, mid_headx, mid_heady, tail_x_10, tail_y_10):
    ##Fin angle computatright
    #Fin angle computatright
    right_fin_tip_x =  df['right_fin_tip'].values[:, 0].astype('float')
    right_fin_tip_y =  df['right_fin_tip'].values[:, 1].astype('float')
    right_fin_base_x =  df['right_fin_base'].values[:, 0].astype('float')
    right_fin_base_y =  df['right_fin_base'].values[:, 1].astype('float')

    left_fin_tip_x =  df['left_fin_tip'].values[:, 0].astype('float')
    left_fin_tip_y =  df['left_fin_tip'].values[:, 1].astype('float')
    left_fin_base_x =   df['left_fin_base'].values[:, 0].astype('float')
    left_fin_base_y =   df['left_fin_base'].values[:, 1].astype('float')

    # lets make all the vectors
    a = left_fin_base_x-left_fin_tip_x
    b = left_fin_base_y-left_fin_tip_y
    left_fin_vect = np.array([b,-a])

    a = right_fin_base_x-right_fin_tip_x 
    b = right_fin_base_y-right_fin_tip_y
    right_fin_vect = np.array([-b,a])

#     mid_headx, mid_heady = midpoint(left_mid_eye_x,left_mid_eye_y, right_mid_eye_x, right_mid_eye_y) #xy left, xy right
    body_vect = np.vstack((mid_headx -tail_x_10[0,:] , mid_heady - tail_y_10[0,:])) 

    ## Compute angles between vectors
    left_fin_angle =  compute_angle_between_vect(left_fin_vect, body_vect)
    right_fin_angle =  compute_angle_between_vect(right_fin_vect, body_vect)

    #nan movement artifacts
    left_fin_angle = left_fin_angle - left_fin_angle[0]
    right_fin_angle = right_fin_angle - right_fin_angle[0]
    left_fin_angle[abs(np.diff(left_fin_angle, prepend=[0])) >= 2] = 0 #np.nan #np.pi
    right_fin_angle[abs(np.diff(right_fin_angle, prepend=[0])) >= 2] = 0 #np.nan #np.pi
    
    # left_fin_angle[abs(left_fin_angle) >= np.pi] = 0 #np.nan #np.pi
    # right_fin_angle[abs(right_fin_angle)  >= np.pi] = 0 #np.nan #np.pi
    print ('corr fins')

    return left_fin_vect, right_fin_vect, left_fin_angle, right_fin_angle

def compute_body_angle(head_x, head_y, body_x, body_y):
    """
    Computes the angle between two points in 2D space.
    
    Parameters:
    head_x, head_y: Coordinates of the first point (head).
    body_x, body_y: Coordinates of the second point (body).
    
    Returns:
    angles_radians: The angle in radians.
    angles_degrees: The angle in degrees.
    """
    # Calculate the differences in the x and y coordinates
    delta_x = body_x - head_x
    delta_y = body_y - head_y

    # Calculate the angle using numpy's arctan2
    angles_radians = np.arctan2(delta_y, delta_x)

    # Convert the angle from radians to degrees
    angles_degrees = np.degrees(angles_radians)

    return angles_radians, angles_degrees



from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colors
import matplotlib.pyplot as plt

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


# Load the Set2 colormap
cmap = plt.get_cmap('Set2')

# Get two colors from the colormap
color1 = cmap(0)  # Get the first color
color2 = cmap(3)  # Get the second color
color3 = cmap(7)  # Get the second color

# print("Color 1:", color1)
# print("Color 2:", color2)

color_ipsi_cont = [ color1, color2, color3]