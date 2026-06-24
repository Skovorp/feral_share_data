import numpy as np
import copy
def init_nan_matrix(mat_shape):
    nan_mat = np.zeros(mat_shape)
    nan_mat[:] = np.nan
    return nan_mat

def apply_function_to_obj_list(obj_list, function_to_apply):
    new_obj_list = []
    for obj in obj_list: 
        new_obj_list.append(function_to_apply(obj))
    return new_obj_list
# def apply_function_to_obj_list(obj_list, function_to_apply, **args):
#     new_obj_list = []
#     for obj in obj_list: 
#         print("args", args)
#         new_obj_list.append(function_to_apply(obj, args))
#     return new_obj_list
##https://stackoverflow.com/questions/803616/passing-functions-with-arguments-to-another-function-in-python
##https://stackoverflow.com/questions/8954746/python-arguments-as-a-dictionary

def get_first_and_last_nan_frames_of_matrix(arr, axis = 1, row_i = 1):
    # Find first and last NaN column indices for each row
    nan_first_col = np.argmax(np.isnan(arr), axis=axis)
    nan_last_col = arr.shape[row_i] - np.argmax(np.flip(np.isnan(arr), axis=axis), axis=axis) - 1
    
    all_nonnan_rows = np.argwhere(np.all(~np.isnan(arr),axis = axis)).flatten()
    nan_first_col[all_nonnan_rows] = 0
    nan_last_col[all_nonnan_rows] = 0
    return nan_first_col, nan_last_col


def get_first_and_last_non_nan_frames_of_matrix(arr, axis =1,row_i = 1):
    # Find first and last NaN column indices for each row
    first_nonnan_col = np.argmax(~np.isnan(arr), axis=axis)
    last_nonnan_col = arr.shape[row_i] - np.argmax(np.flip(~np.isnan(arr), axis=axis), axis=axis) - 1
    all_nan_rows = np.argwhere(np.all(np.isnan(arr),axis = axis)).flatten()
    first_nonnan_col[all_nan_rows] = 0
    last_nonnan_col[all_nan_rows] = 0
    
    return first_nonnan_col, last_nonnan_col

def reshape_matrices_in_list(mat_list, shape_0, shape_1):
    reshaped_mats = []
    
    for mat in mat_list:
        mat_shape = list(mat.shape)
        mat_shape[0:2] = [shape_0, shape_1]
        
        reshaped_mats.append(mat.reshape(mat_shape, order = 'F'))
    return reshaped_mats

def get_mean_for_matrices_in_list(mat_list, dim = 1):
    mean_mats = []
    for mat in mat_list:
        
        mean_mats.append(np.nanmean(mat, axis = dim))
    return mean_mats

def get_std_for_matrices_in_list(mat_list, dim = 1):
    mean_mats = []
    for mat in mat_list:
        mean_mats.append(np.nanstd(mat, axis = dim))
    return mean_mats
def crop_to_specific_frames_in_mat_list(mat_list, frame_start, frame_end):
    filtered_mats = []
    
    for mat in mat_list:
        filtered_mats.append(mat[frame_start:frame_end,:])
        
    return filtered_mats

# def crop_to_specific_frames_in_mat_list(mat_list, frames_to_make_nans, features_to_index, feature_to_skip={"trackID", "frame_time", "plate_num"}):
#     filtered_mats = []
#     mat_list = copy.deepcopy(mat_list)
    
#     for i in range(mat_list):
#         mat = mat_list[i]
#         if features_to_index[i] in feature_to_skip:
#             filtered_mats.append(mat)
#             continue
#         mat[frames_to_make_nans,:] = np.nan
#         filtered_mats.append(mat)
        
#     return filtered_mats
    
def change_indices_to_nans_in_mat_list(mat_list,nan_indices, exception_is = {}):
    nan_indices_x, nan_indices_y = np.where(nan_indices)
    mat_list = update_dtype_in_mat_list(mat_list, 'float64', exception_is = exception_is)
    filtered_mats = []
    for mat_i in range(len(mat_list)):
        mat = mat_list[mat_i]
        new_mat = copy.deepcopy(mat)
        if mat_i in exception_is:
            filtered_mats.append(new_mat)
            continue

        new_mat[nan_indices_x,nan_indices_y ] = np.nan
        filtered_mats.append(new_mat)
    #filtered_mats = update_dtype_in_mat_list(mat_list, 'float64')

    return filtered_mats
def change_frames_to_nans_in_mat_list(mat_list,frame_start, frame_end, exception_is = {}):
    # nan_indices_x, nan_indices_y = np.where(nan_indices)
    mat_list = update_dtype_in_mat_list(mat_list, 'float64', exception_is = exception_is)
    filtered_mats = []
    for mat_i in range(len(mat_list)):
        mat = mat_list[mat_i]
        new_mat = copy.deepcopy(mat)
        if mat_i in exception_is:
            filtered_mats.append(new_mat)
            continue

        new_mat[frame_start:frame_end ] = np.nan
        filtered_mats.append(new_mat)
    #filtered_mats = update_dtype_in_mat_list(mat_list, 'float64')

    return filtered_mats

    
def filter_tracks_in_mat_list(mat_list, tracks_of_interest):
    filtered_mats = []
    mat_list = copy.deepcopy(mat_list)
    for mat in mat_list:
        if len(mat)==0:
            continue
        if tracks_of_interest.size==0:
            empty_mat = np.empty((mat.shape[0], 0 ), dtype = mat.dtype)
            filtered_mats.append(empty_mat)
        else:
            filtered_mats.append(mat[:,tracks_of_interest ])

            
            
            
        
    return filtered_mats

def filter_array_list_for_tracks(array_list, tracks):
    filtered_array_list = []
    for arr in array_list:
        filtered_array_list.append(arr[tracks])
    return filtered_array_list
    
def update_dtype_in_mat_list(mat_list, dtype, exception_is = {} ):
    filtered_mats = []
    
    for mat_i in range(len(mat_list)):
        if mat_i in exception_is:
            filtered_mats.append(mat)
            continue
        mat = mat_list[mat_i]
        filtered_mats.append(mat.astype(dtype))
        
    return filtered_mats

def append_track_mat_to_mat_list(mat_list, track_mat):
    mat_list.append(track_mat)
    return mat_list

def append_corresponding_track_mat_to_mat_list(mat_list, track_mat):
    for i in range(len(mat_list)):
        mat_list[i].append(track_mat[i])
    #mat_list.append(track_mat)
    return mat_list
def init_track_mat(mat_list):
    new_mat_list = []
    for i in range(len(mat_list)):
        init_mat_list_shape = list(mat_list[i].shape)
        init_mat_list_shape[1] = 0
        new_mat_list.append(np.empty(init_mat_list_shape, dtype = mat_list[i].dtype))
    #mat_list.append(track_mat)
    return new_mat_list

def concatenate_corresponding_track_mat_to_mat_list(mat_list, track_mat):
    mat_list = copy.deepcopy(mat_list)
    track_mat = copy.deepcopy(track_mat)
    for i in range(len(mat_list)):
        
        mat_list[i] = np.concatenate([mat_list[i], track_mat[i]], axis=1)
    #mat_list.append(track_mat)
    return mat_list


def get_repeated_col(a):
    repeated_indices = []
    unq, count = np.unique(a.T, axis=0, return_counts=True)
    repeated_groups = unq[count > 1]

    for repeated_group in repeated_groups:
        repeated_idx = np.argwhere(np.all(a.T == repeated_group, axis=1))
        #print(repeated_idx.ravel())
        repeated_indices.append(repeated_idx.ravel())
    return repeated_indices

def get_all_track_start_and_end_is_from_binary_event_matrix(event_bin_mat, inverted_mat = False):
    event_bin_mat = copy.deepcopy(event_bin_mat)
    event_bin_mat[np.isnan(event_bin_mat)]=0
    n_tracks = event_bin_mat.shape[1]
    if inverted_mat:
        event_bin_mat_buffered = np.vstack((np.ones((1, n_tracks)), event_bin_mat, np.ones((1, n_tracks))))
    else:
        event_bin_mat_buffered = np.vstack((np.zeros((1, n_tracks)), event_bin_mat, np.zeros((1, n_tracks))))
    event_boundaries = np.diff(event_bin_mat_buffered.T, axis = 1)
    event_starts = np.argwhere(event_boundaries==1)
    event_ends = np.argwhere(event_boundaries ==-1)
    return event_starts, event_ends

def multirange(start_indices, end_indices, fsize):
    result = np.zeros(fsize, dtype=int)
    start_indices = start_indices.flatten().astype(np.int32)
    end_indices = end_indices.flatten().astype(np.int32)
    for i in range(start_indices.shape[0]):
        start = start_indices[i]
        end = end_indices[i]
        result[start:end] = 1
    return result


if __name__ == "__main__":
    arr = np.array([[1, np.nan, 3, np.nan,4],
                [np.nan, 5, 6, np.nan, np.nan],
                [7, 10,  np.nan, np.nan, np.nan], 
                [np.nan, np.nan,  np.nan, np.nan, np.nan],
                [1,1, 1, 1, 1]])
    print(arr)
    nan_first_col, nan_last_col = get_first_and_last_nan_frames_of_matrix(arr)
    print(nan_first_col, nan_last_col)

    # first_nonnan_col = np.argmax(~np.isnan(arr), axis=1)
    # last_nonnan_col = arr.shape[1] - np.argmax(~np.flip(~np.isnan(arr), axis=1), axis=1) - 1

    
    first_nonnan_col, last_nonnan_col = get_first_and_last_non_nan_frames_of_matrix(arr) #index of last non an 
    print(first_nonnan_col, last_nonnan_col)