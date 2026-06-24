
def filter_tracks_in_dict(dictionary, tracks_of_interest):
    filtered_dict = {
        key: value[:,tracks_of_interest ] for key, value in dictionary.items()
    }

    return filtered_dict

def update_dtype_in_dict(dictionary, dtype, exceptions = {} ):
    #filtered_dict = {}
    filtered_dict_exceptions = {key: value for key, value in dictionary.items() if key in exceptions}
    filtered_dict = {key: value.astype(dtype) for key, value in dictionary.items() if key not in exceptions}
    filtered_dict.update(filtered_dict_exceptions)
    return filtered_dict

def filter_array_dict_for_tracks(dictionary, tracks_of_interest):
    filtered_dict = {
        key: value[tracks_of_interest ] for key, value in dictionary.items()
    }

    return filtered_dict
    # for key, value in dict.items():
    #     if key in exceptions:
    #         filtered_dict[key] = 
    #         continue
    #     mat = mat_list[mat_i]:
    #     filtered_mats.append(mat.astype(dtype))
        
    # return filtered_mats