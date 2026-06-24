import numpy as np
from scipy import interpolate
import scipy.spatial.distance
import itertools
import pandas as pd
import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt




def get_curve_length(ordered_curve_coords):
    return np.sum(np.linalg.norm(np.diff(ordered_curve_coords, axis = 0),axis=1))

def get_even_points_along_curve(ordered_curve_coords, n_pts_to_sample = 12): #get_even_points_along_side
    n_pts = ordered_curve_coords.shape[0]


    diff_pts = np.diff(ordered_curve_coords, axis = 0 )
    dst_pts = np.cumsum(np.linalg.norm(diff_pts, axis = 1), axis = 0 )
    dst_pts = np.concatenate((np.asarray([0]), dst_pts), axis = 0)

    total_worm_length = dst_pts[-1]
    evenly_spaced_dst = np.linspace(0,total_worm_length, num = n_pts_to_sample)


    #dst_f= interpolate.interpol1d(np.arange(0,n_pts),dst_pts)
    dst_finv = interpolate.interp1d(dst_pts, np.arange(0,n_pts))

    evenly_spaced_is = (dst_finv(evenly_spaced_dst)).astype('int16')
    #todo could laso take points in between but i think not worth
    evenly_spaced_pts = ordered_curve_coords[evenly_spaced_is,:]

    return evenly_spaced_pts
def interpolate_data_at_timepoints(data, evenly_spaced_times, frame_times):#actual_times_csv):
    
#        frame_times = pd.read_csv(actual_times_csv, sep=',', usecols=['frame_time']).to_numpy().flatten()

    interpolated_data = interpolate.griddata(frame_times, data,evenly_spaced_times, method='linear')
    return interpolated_data

def get_segment_coords_between_two_indices(first_i, second_i, ordered_curve_coords):#get_side_coords_from_contour
    if first_i==second_i:
        print("in get_segment_coords_between_two_indices in NPCurvehandler")
        print("first_i==second_i")
        print(first_i, second_i)
        return ordered_curve_coords
    if first_i<second_i:
        segment_1 = ordered_curve_coords[first_i:second_i,:]
        segment_2 = np.flip(np.concatenate((ordered_curve_coords[second_i:,:], ordered_curve_coords[:first_i,:]), axis = 0), axis = 0)
        #return segment_1, segment_2
    elif first_i>second_i:#can also np flip if cared about ventral/dorsal but dont atm i think...
        segment_1 = np.concatenate((ordered_curve_coords[first_i:,:], ordered_curve_coords[:second_i,:]), axis = 0)
        segment_2 = np.flip(ordered_curve_coords[second_i:first_i,:], axis =0)
    return segment_1, segment_2
def find_nearest_coordinate_pair_in_list(coord_pair, coords_list):
    return np.argmin(np.linalg.norm(coords_list-coord_pair, axis = 1))

def get_evenly_spaced_timepoints_along_array(org_frame_times, fps):
    #org_fraem times is actal frame time recordd; want times to smaple in between that are evenly spaced and within the min /max bounds of the org_frame_times
    min_time = np.min(org_frame_times)
    min_time = round(min_time*fps)/fps
    max_time = np.max(org_frame_times)
    #max_time = np.max(frame_times)
    max_time = round(max_time*fps)/fps
    evenly_spaced_timepoints = np.arange(min_time, max_time, 1/fps)
    return evenly_spaced_timepoints

def calculate_curvature(points):
    # Calculate the first derivative
    first_derivative = np.gradient(points, axis=0)
    
    # Calculate the second derivative
    second_derivative = np.gradient(first_derivative, axis=0)
    
    # Calculate the curvature
    numerator = np.abs(first_derivative[:, 0] * second_derivative[:, 1] -
                    first_derivative[:, 1] * second_derivative[:, 0])
    denominator = np.power(np.square(first_derivative[:, 0]) + np.square(first_derivative[:, 1]), 1.5)
    
    curvature = numerator / denominator
    
    return curvature
        

def dot_product(vec1,vec2, decimals_to_round_to = 3 ):
    y1 = vec1[:,1]
    x1 = vec1[:,0]

    y2 = vec2[:,1]
    x2 = vec2[:,0]

    dot = x1*x2+y1*y2
    dot = np.around(dot, decimals_to_round_to)
    return dot

def get_signed_angle_between_norm_vectors(vec1, vec2):
    #print(vec1.shape, "vec1.shape")
    #print(vec2.shape, "vec2.shape")
    dot = dot_product(vec1,vec2)
    angle = np.arccos(dot)
    cross = np.cross(vec1,vec2, axis = 1)
    change_sign_i = np.where(cross<0)[0]
    angle[change_sign_i] = -angle[change_sign_i]
    return angle

def get_signed_supplementary_angles_between_coords1(coords_list):

    vecs = np.diff(coords_list, axis = 0)
    norm = np.linalg.norm(vecs, axis = 1)[:,None]
    norm_cat = np.concatenate((norm,norm),axis =1)

    vecs = vecs/norm_cat

    vec2 = vecs[1:,:] #startign tt index 1 get ector leavign that index and goinf to adjacent point
    vec1 = vecs[0:-1,:] #startign at index 1 get vectors point to that index

    angle = get_signed_angle_between_norm_vectors(vec2, vec1)
    # dot = dot_product(vec1,vec2)
    # angle = np.arccos(dot);
    # cross = np.cross(vec1,vec2, axis = 1)
    # change_sign_i = np.where(cross<0)[0]
    # angle[change_sign_i] = -angle[change_sign_i]
    return angle

def convert_radians_to_degrees(radians_vec):
    return np.degrees(radians_vec)

def get_two_furthest_pts_index( np_pts_list):
    #def option2(r):
    dists = scipy.spatial.distance.pdist(np_pts_list, 'cityblock')
    dist_i = np.argmax(dists)
    n_extrema = np_pts_list.shape[0]
    pt_i1, pt_i2 = recover_indices_from_scipy_condensed_matrix_index(dist_i, n_extrema)
    return pt_i1, pt_i2

def recover_indices_from_scipy_condensed_matrix_index( i, n_nodes):
    pairs = list(itertools.combinations(range(n_nodes),2))
    return pairs[i]

# def get_start_end_is_of_ones_in_binary_array( binary_array):
#     '''
#     get frames in which consecutive chunk of good midlines or a "midline_chunk" starts and ends
#     '''
#     binary_array_buffered = np.concatenate([np.zeros(1),binary_array, np.zeros(1)])
#     chunk_start_is = np.argwhere(np.diff(binary_array_buffered)==1).flatten()
#     chunk_end_is = np.argwhere(np.diff(binary_array_buffered)==-1).flatten()

#     return chunk_start_is, chunk_end_is


def get_start_end_is_of_ones_in_binary_array( binary_array):
    '''
    get frames in which consecutive chunk of good midlines or a "midline_chunk" starts and ends
    '''
    first_track_frame = np.min(np.argwhere(~np.isnan(binary_array)))
    first_track_frame = np.max((0, first_track_frame))
    last_track_frame = np.max(np.argwhere(~np.isnan(binary_array)))
    n_frames = binary_array.shape[0]
    last_track_frame = np.min((last_track_frame+1, n_frames))
    
    binary_array_buffered = np.concatenate([np.zeros(1),binary_array, np.zeros(1)])
    binary_array_buffered[first_track_frame] = 0 
    binary_array_buffered[last_track_frame] = 0 
    
    chunk_start_is = np.argwhere(np.diff(binary_array_buffered)==1).flatten()
    chunk_end_is = np.argwhere(np.diff(binary_array_buffered)==-1).flatten()

    return chunk_start_is, chunk_end_is
# def get_start_end_is_of_ones_in_binary_array( binary_array):
#     # is_nan_bin = ~np.isnan(binary_array).flatten().astype('int8')
#     is_nan_bin = np.logical_not(np.isnan(binary_array).flatten()).astype('int8')
#     isnan_buffered = np.concatenate([np.zeros(1),is_nan_bin, np.zeros(1)])
#     is_nan_start_is = np.argwhere(np.diff(isnan_buffered)==1).flatten()
#     is_nan_start_is[is_nan_start_is < 0] = 0
#     is_nan_end_is = np.argwhere(np.diff(isnan_buffered)==-1).flatten()
#     n_frames = binary_array.shape[0]
#     is_nan_end_is[(is_nan_end_is+1)>n_frames] = n_frames

#     binary_array_buffered = np.concatenate([np.zeros(1),binary_array, np.zeros(1)])
#     binary_array_buffered[is_nan_start_is] = 0 
#     binary_array_buffered[is_nan_end_is] = 0 

#     chunk_start_is = np.argwhere(np.diff(binary_array_buffered)==1).flatten()
#     chunk_end_is = np.argwhere(np.diff(binary_array_buffered)==-1).flatten()

#     return chunk_start_is, chunk_end_is
# def get_start_end_is_of_ones_in_binary_array( binary_array):
#     # is_nan_bin = ~np.isnan(binary_array).flatten().astype('int8')
#     is_nan_bin = np.logical_not(np.isnan(binary_array).flatten()).astype('int8')
#     isnan_buffered = np.concatenate([np.zeros(1),is_nan_bin, np.zeros(1)])
#     is_nan_start_is = np.argwhere(np.diff(isnan_buffered)==1).flatten()
#     is_nan_start_is[is_nan_start_is < 0] = 0
#     is_nan_end_is = np.argwhere(np.diff(isnan_buffered)==-1).flatten()
#     n_frames = binary_array.shape[0]
#     is_nan_end_is[(is_nan_end_is+1)>n_frames] = n_frames

#     binary_array_buffered = np.concatenate([np.zeros(1),binary_array, np.zeros(1)])
#     binary_array_buffered[is_nan_start_is] = 0 
#     binary_array_buffered[is_nan_end_is] = 0 

#     chunk_start_is = np.argwhere(np.diff(binary_array_buffered)==1).flatten()
#     chunk_end_is = np.argwhere(np.diff(binary_array_buffered)==-1).flatten()

#     return chunk_start_is, chunk_end_is

import copy 
def get_start_end_is_of_ones_in_binary_array( binary_array):
    '''
    get frames in which consecutive chunk of good midlines or a "midline_chunk" starts and ends
    '''
    binary_array = copy.deepcopy(binary_array.astype('int'))
    binary_array[np.isnan(binary_array)] = 0
    binary_array_buffered = np.concatenate([np.zeros(1),binary_array, np.zeros(1)])
    chunk_start_is = np.argwhere(np.diff(binary_array_buffered)==1).flatten()
    chunk_end_is = np.argwhere(np.diff(binary_array_buffered)==-1).flatten()

    return chunk_start_is, chunk_end_is
def get_durations_of_ones_in_binary_array( chunk_start_is, chunk_end_is):
    return chunk_end_is-chunk_start_is

def get_curve_length(array):
    # Calculate the Euclidean distance between consecutive points
    pairwise_diff = np.diff(array, axis=0)
    distances = np.linalg.norm(pairwise_diff, axis=1)
    
    # Summing up the distances to get the total length
    length = np.sum(distances)
    return length

def get_midline_length_from_ordered_coords( ordered_midline_coords):
    midline_length = np.sum(np.linalg.norm(np.diff(ordered_midline_coords, axis = 0),axis=1))
    return midline_length

def create_spline1(points, n_spline_pts, smoothness = 0 ):
    # Convert the points list into separate X and Y arrays
    points = np.array(points)
    x = points[:, 0]
    y = points[:, 1]

    # Perform spline interpolation
    tck, u = splprep([x, y], s=smoothness)

    # Evaluate the spline on a finer parameterization
    u_fine = np.linspace(0, 1, num=n_spline_pts)
    spline_x, spline_y = splev(u_fine, tck)

    # # Plot the original points and the spline
    # plt.plot(x, y, 'ro', label='Original Points')
    # plt.plot(spline_x, spline_y, 'b-', label='Spline')
    # plt.legend()
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.show()
    return spline_x, spline_y, tck

def get_curvature_from_spline(tck, n_total_spline_pts = 33):
    u_fine = np.linspace(0, 1, num=n_total_spline_pts)
    # - https://stackoverflow.com/questions/13590989/peak-curvature-in-scipy-spline
    # https://en.wikipedia.org/wiki/Curvature
    
    
    curvatures = splev(u_fine, tck, der = 2)
    return curvatures

    
