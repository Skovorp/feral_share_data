import matplotlib.pyplot as plt
import numpy as np
import cv2

from GenerateBehTrack.WormShape.Worm_Midline import Worm_Midline

from VectorandImageHandlers.NPCurveHandler import NPCurveHandler
from VectorandImageHandlers.DebugImgHandler import DebugImgHandler
from VectorandImageHandlers.BinaryImageHandler import BinaryImageHandler

from FileHandlers.YamlHandler import YamlHandler

class WormMorphology():
    def __init__(self, frame_i, bbox_img, params_yaml):

        self.frame_i = frame_i
        self.bbox_img = bbox_img

        self.thresholded_and_closed_worm = np.asarray([])

        self.outer_worm_contour = np.asarray([])
        self.sorted_worm_contours = []
        self.worm_mask = np.asarray([])
        self.worm_midline_coords = np.asarray([])
        self.extrema = np.ones((2,2))*-1
        self.is_looping = True
        self.good_coords = False

        self.worm_midline = Worm_Midline(params_yaml)
        self.BinaryImageHandler = BinaryImageHandler()

        self.params_yaml = params_yaml
        yaml_handler = YamlHandler()
        param_dict = yaml_handler.get_dictionary_from_yaml(params_yaml)["worm_morph_params"]


        self.brightness_threshold = param_dict['brightness_threshold']

        closing_kernel_size = param_dict['closing_kernel_size']
        self.closing_kernel = np.ones((closing_kernel_size, closing_kernel_size), dtype = 'uint8')

        # erosion_kernel_size = param_dict['erosion_kernel_size']
        # self.erosion_kernel = np.ones((erosion_kernel_size, erosion_kernel_size), dtype = 'uint8')
        self.erosion_kernel = np.asarray([[0,1,0], [1,1,1], [0,1,0]], dtype = 'uint8')

        self.area_to_perimeter = param_dict['area_to_perimeter']

        self.area_to_perimeter_ratio_thres = param_dict['area_to_perimeter_ratio_thres']


    def get_outer_worm_contour(self):
        if len(self.sorted_worm_contours)>0:
            self.outer_worm_contour = self.sorted_worm_contours[0]
            return self.outer_worm_contour

        if self.thresholded_and_closed_worm.size == 0:
            self.thresholded_and_closed_worm = self.BinaryImageHandler.threshold_and_close_img(self.bbox_img, self.brightness_threshold, self.closing_kernel)

        self.outer_worm_contour = self.BinaryImageHandler.get_largest_contour_from_binary_img(self.thresholded_and_closed_worm)
        return self.outer_worm_contour

    def get_sorted_worm_contours(self):
        if self.thresholded_and_closed_worm.size == 0:
            self.thresholded_and_closed_worm = self.BinaryImageHandler.threshold_and_close_img(self.bbox_img, self.brightness_threshold, self.closing_kernel)

        sorted_contours = self.BinaryImageHandler.get_contours_sorted_by_n_pts(self.thresholded_and_closed_worm)

        self.outer_worm_contour = sorted_contours[0]

        is_looping =  self.detect_loop()

        if not is_looping:
            self.sorted_worm_contours = [sorted_contours[0]]
        else:
            self.sorted_worm_contours = sorted_contours[0:2]

        return self.sorted_worm_contours

    def get_worm_mask(self):
        if len(self.sorted_worm_contours) == 0:
            self.get_sorted_worm_contours()

        if self.worm_mask.size == 0:
            img_shape = self.bbox_img.shape
            self.worm_mask = self.BinaryImageHandler.draw_img_mask_from_contours(img_shape, self.sorted_worm_contours)

        return self.worm_mask

    def get_worm_centroid(self):
        if self.worm_mask.size == 0:
            self.get_worm_mask()

        worm_centroid = self.BinaryImageHandler.get_centroid_from_binary_img(self.worm_mask)
        return worm_centroid

    def detect_loop(self,ratio_thres=1.15):

        if self.outer_worm_contour.size == 0:
            self.get_outer_worm_contour()

        if cv2.contourArea(self.outer_worm_contour)/cv2.arcLength(self.outer_worm_contour,True)>self.area_to_perimeter*self.area_to_perimeter_ratio_thres:

            return True
        else:
            # print("not looping")
            # print("ratio", cv2.contourArea(self.outer_worm_contour)/cv2.arcLength(self.outer_worm_contour,True))
            return False

    def get_midline(self):
        if self.worm_mask.size == 0:
            self.get_worm_mask()
            #print("please get worm_mask")
        self.good_coords, self.worm_midline_coords, n_extrema, skel_extrema_is , skel_extrema_pts = self.worm_midline.get_midline_from_worm_mask(self.worm_mask)

        self.worm_midline_coords = self.worm_midline.sample_midline_coords_evenly()

        return self.good_coords, self.worm_midline_coords

    def get_midline_length(self):
        if self.worm_midline_coords.shape==0:
            print("warning: worm_midline not yet determined")
        midline_length = self.worm_midline.get_length_from_midline()

    def get_inner_worm_points(self):
        if self.worm_mask.size == 0:
            self.get_worm_mask()
        eroded_img = cv2.erode(self.worm_mask, self.erosion_kernel,iterations=1)
        inner_worm_points_erosion = np.argwhere(eroded_img)
        inner_worm_points_erosion = inner_worm_points_erosion[:,[1,0]]
        return inner_worm_points_erosion

    def erode_bin_img(self):
        eroded_img = cv2.erode(self.worm_mask, self.erosion_kernel,iterations=1)
        return eroded_img

    def get_debug_image(self):
        # debug_img_handler = DebugImgHandler(self.bbox_img)
        # gray = (200,200,200)
        # worm_mask = self.get_worm_mask()
        # debug_image = debug_img_handler.draw_coords_on_img(self.outer_worm_contour, worm_mask, coord_radius=1, coord_color = gray, coord_thickness = -1)

        # return debug_image
        pass
