
import numpy as np
import scipy.spatial as spatial

from FileHandlers.YamlHandler import YamlHandler

class WormExtremaSectionMeanBrightness():


    def __init__(self, inner_worm_coords, midline, img, params_yaml):
        self.img = img
        self.inner_worm_coords = inner_worm_coords
        self.point_tree = spatial.cKDTree(self.inner_worm_coords)
        self.midline = midline
        self.extrema1_coords = self.midline[0,:]
        self.extrema2_coords = self.midline[-1,:]

        yaml_handler = YamlHandler()
        param_dict = yaml_handler.get_dictionary_from_yaml(params_yaml)["worm_section_brightness_params"]

        self.worm_radii = param_dict["worm_radii"]
        self.n_extrema = 2

    def get_worm_coords_around_extrema(self, cur_extrema_coord):
        pt_is_around_extrema = self.point_tree.query_ball_point(cur_extrema_coord, self.worm_radii)
        worm_coords_around_extrema = self.inner_worm_coords[pt_is_around_extrema,:]
        return worm_coords_around_extrema

    def get_mean_brightness_around_single_extrema(self, worm_coords_around_extrema):

        brightness_values_around_extrema = self.img[worm_coords_around_extrema[:,1], worm_coords_around_extrema[:,0]]
        mean_section_brightness = np.mean(brightness_values_around_extrema)

        return mean_section_brightness

    def get_mean_pixel_brightness_around_both_extrema(self):


        extrema_coords = [self.extrema1_coords, self.extrema2_coords]

        sum_mean_brightness = np.zeros(self.n_extrema)

        for extrema_i in range(self.n_extrema):
            cur_extrema_coord = extrema_coords[extrema_i]
            worm_coords_around_extrema = self.get_worm_coords_around_extrema(cur_extrema_coord)
            mean_brightness = self.get_mean_brightness_around_single_extrema(worm_coords_around_extrema)

            sum_mean_brightness[extrema_i]+=mean_brightness

        return sum_mean_brightness