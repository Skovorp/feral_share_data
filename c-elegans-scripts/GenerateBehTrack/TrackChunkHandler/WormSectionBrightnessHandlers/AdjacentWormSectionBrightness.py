import copy
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(os.path.join("..", "..")))
sys.path.append(os.path.dirname(SCRIPT_DIR))

SCRIPT_DIR = os.path.dirname(os.path.abspath(os.path.join("..", "..","VectorandImageHandlers")))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import scipy.spatial as spatial

from VectorandImageHandlers.NPCurveHandler import NPCurveHandler

from VectorandImageHandlers.DebugImgHandler import DebugImgHandler
from FileHandlers.YamlHandler import YamlHandler


class AdjacentWormSectionMedianBrightness(): #made of a bunch of midlines

    def __init__(self, img, inner_worm_coords, midline, params_yaml, debug_img_name = ""):


        self.n_extrema = 2
        self.n_sections = 2

        self.img = img
        self.inner_worm_coords = inner_worm_coords
        self.point_tree = spatial.cKDTree(self.inner_worm_coords)
        self.midline = midline

        self.debug_img = copy.deepcopy(img)
        self.DebugImgHandler = DebugImgHandler(params_yaml)

        yaml_handler = YamlHandler()
        param_dict = yaml_handler.get_dictionary_from_yaml(params_yaml)["worm_section_brightness_params"]

        self.worm_radii = param_dict["worm_radii"]

        extrema1_section_is_x = param_dict["extrema1_section_is_x"]
        extrema1_section_is_y = param_dict["extrema1_section_is_y"]
        self.extrema1_section_coords = np.asarray([extrema1_section_is_x,extrema1_section_is_y])

        extrema2_section_is_x = param_dict["extrema2_section_is_x"]
        extrema2_section_is_y = param_dict["extrema2_section_is_y"]
        self.extrema2_section_coords = np.asarray([extrema2_section_is_x,extrema2_section_is_y])


        self.section_colors = [[0,0,0], [150,150,150]]

    def set_extrema_section_coords(self, extrema1_section_coords, extrema2_section_coords):
        self.extrema1_section_coords = extrema1_section_coords
        self.extrema2_section_coords = extrema2_section_coords


    def compare_median_brightness_difference_of_adjacent_sections_around_extremas(self):
        #compare brightness between sections to determine which side is head

        midline_coords_near_each_extrema = self.get_midline_coords_near_each_extrema()

        median_brightness_differences = np.zeros(self.n_extrema)

        for extrema_i in range(self.n_extrema):
            midline_coords_near_given_extrema = midline_coords_near_each_extrema[extrema_i]
            section_median_brightness_difference = self.get_section_median_brightness_difference(midline_coords_near_given_extrema)
            median_brightness_differences[extrema_i] = section_median_brightness_difference
            # section_median_brightness = np.zeros(n_sections)

            # for section_i in range(self.n_sections):
            #     #section_worm_pt_is = both_section_worm_pt_is[section_i]
            #     #section_worm_coords = inner_worm_coords[section_worm_pt_is,:]
            #     section_worm_coords = both_section_worm_coords[section_i]
            #     section_median_brightness[section_i] = self.get_section_median_pixel_brightness(section_worm_coords)

            #     self.update_debug_image(self,section_i, section_worm_coords

        head_i = np.argmin(median_brightness_differences)

        return head_i

    def get_midline_coords_near_each_extrema(self):
        midline_coords_near_extrema_1 = self.midline[self.extrema1_section_coords,:]
        midline_coords_near_extrema_2 = self.midline[self.extrema2_section_coords,:]#np.flip(side_evenly_spaced_pts[-3:-1,:], axis =0)#CHECK

        midline_coords_near_each_extrema = [midline_coords_near_extrema_1, midline_coords_near_extrema_2]
        return midline_coords_near_each_extrema

    def get_adjacent_sections_worm_coords(self, midline_coords_near_given_extrema ):
        section1_evenly_spaced_pts = midline_coords_near_given_extrema[0,:]
        section2_evenly_spaced_pts = midline_coords_near_given_extrema[1,:]

        section1_worm_pt_is = self.point_tree.query_ball_point(section1_evenly_spaced_pts, self.worm_radii)
        section2_worm_pt_is = self.point_tree.query_ball_point(section2_evenly_spaced_pts, self.worm_radii)

        if not np.setdiff1d(section2_worm_pt_is, section1_worm_pt_is).size ==0:
            section2_worm_pt_is = np.setdiff1d(section2_worm_pt_is, section1_worm_pt_is)

        #both_section_worm_pt_is = [section1_worm_pt_is, section2_worm_pt_is]
        both_section_worm_coords = [self.inner_worm_coords[section1_worm_pt_is,:], self.inner_worm_coords[section2_worm_pt_is,:]]

        return both_section_worm_coords

    def get_section_median_brightness_difference(self, midline_coords_near_given_extrema):
        adjacent_sections_worm_coords = self.get_adjacent_sections_worm_coords(midline_coords_near_given_extrema)
        section_median_brightness_difference = self.get_section_median_brightness_difference_from_worm_coords(adjacent_sections_worm_coords)
        return section_median_brightness_difference

    def get_section_median_brightness_difference_from_worm_coords(self, both_section_worm_coords):

        section_median_brightness = np.zeros(self.n_sections)

        for section_i in range(self.n_sections):
            #section_worm_pt_is = both_section_worm_pt_is[section_i]
            #section_worm_coords = inner_worm_coords[section_worm_pt_is,:]
            section_worm_coords = both_section_worm_coords[section_i]
            #self.update_debug_image(section_i, section_worm_coords)

            section_median_brightness[section_i] = self.get_section_median_pixel_brightness(section_worm_coords)

        median_brightness_difference  = np.diff(section_median_brightness)
        return median_brightness_difference

    def get_section_median_pixel_brightness(self, section_worm_coords):
        section_px_values = self.img[section_worm_coords[:,1], section_worm_coords[:,0]]
        return np.median(section_px_values)

    def update_debug_image(self,section_i, section_worm_coords):
        section_color = self.section_colors[section_i]
        self.debug_img = self.DebugImgHandler.draw_coords_on_img(section_worm_coords, self.debug_img, coord_radius=1, coord_color = section_color, coord_thickness = -1)

    def save_debug_image(self, debug_img_name):
        self.DebugImgHandler.save_img(debug_img_name, self.debug_img)

    def get_extrema_section_coords(section_length_frac, midline_coords,worm_radii,  n_midline_pts= 50):
        NPCurveHandler = NPCurveHandler()
        midline_length = NPCurveHandler.get_curve_length(midline_coords)
        evenly_spaced_midline_pts = NPCurveHandler.get_even_points_along_curve(midline_coords, n_pts_to_sample = n_midline_pts)

        n_section_midline_pts = int(n_midline_pts*section_length_frac)
        #account for worm radii
        length_per_midline_pt = midline_length/n_midline_pts
        worm_radii_pts =worm_radii/length_per_midline_pt
        n_section_midline_pts-int(worm_radii_pts)

        extrema1_section1_coords = np.arange(0,n_section_midline_pts)
        extrema1_section2_coords = np.arange(n_section_midline_pts,2*n_section_midline_pts)
        extrema1_section_coords = [extrema1_section1_coords,extrema1_section2_coords ]


        starti = n_midline_pts-n_section_midline_pts
        endi =  n_midline_pts
        extrema2_section1_coords = np.arange(starti,endi)

        endi = n_midline_pts - n_section_midline_pts
        starti =  n_midline_pts- 2*n_section_midline_pts
        extrema2_section2_coords = np.arange(starti,endi)
        extrema2_section_coords = [extrema2_section1_coords,extrema2_section2_coords ]

        return extrema1_section_coords, extrema2_section_coords