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
        self.outer_loop_area_min = param_dict['outer_loop_area_min']
        self.inner_loop_area_min = param_dict['inner_loop_area_min']
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

    def get_sorted_worm_contours(self, bbox_centroid = np.array([50,50])):
        if self.thresholded_and_closed_worm.size == 0:
            self.thresholded_and_closed_worm = self.BinaryImageHandler.threshold_and_close_img(self.bbox_img, self.brightness_threshold, self.closing_kernel)

        #sorted_contours = self.BinaryImageHandler.get_contours_sorted_by_n_pts(self.thresholded_and_closed_worm)

        #self.outer_worm_contour = sorted_contours[0]#### THIS IS THE ISSUE need to get contour closest to center..

        #outer_contours = np.array(self.BinaryImageHandler.get_contours(self.thresholded_and_closed_worm))
        outer_contours = self.BinaryImageHandler.get_contours(self.thresholded_and_closed_worm)
        outer_contour_areas = np.array([cv2.contourArea(outer_contour) for outer_contour in outer_contours])
        large_area_outer_contour_is = np.argwhere(outer_contour_areas>self.outer_loop_area_min).flatten()
        outer_contours = [outer_contours[i] for i in large_area_outer_contour_is]#outer_contours[outer_contour_areas>self.outer_loop_area_min]
        #if outer_contours.size ==0:
        if len(outer_contours)==0:
            self.sorted_worm_contours = outer_contours
            return outer_contours
        
        outer_contour_centroids = np.concatenate([self.BinaryImageHandler.get_centroid_from_contour(outer_contour)[None,:] for outer_contour in outer_contours], axis =0)
        outer_contour_dist_from_bbox_centroid = np.linalg.norm(outer_contour_centroids-bbox_centroid, axis = 1) 
        outer_contour_i = np.argmin(outer_contour_dist_from_bbox_centroid) #sorted lowest to highest 
        self.outer_worm_contour = outer_contours[outer_contour_i]

        outer_contour_centroid = outer_contour_centroids[outer_contour_i,:]
        outer_contour_area = cv2.contourArea(self.outer_worm_contour)

        is_looping =  self.detect_loop()

        if not is_looping:
            self.sorted_worm_contours = [self.outer_worm_contour]
        else:
            inner_worm_contour = self.get_inner_worm_closest_to_outer_worm_contour(outer_contour_centroid, outer_contour_area)
            if inner_worm_contour is None:
                self.sorted_worm_contours = [self.outer_worm_contour]
            else:
                self.sorted_worm_contours = [self.outer_worm_contour, inner_worm_contour]

        return self.sorted_worm_contours

    def get_inner_worm_closest_to_outer_worm_contour(self, outer_contour_centroid, outer_contour_area):
        #all_contours = np.array(self.BinaryImageHandler.get_contours(self.thresholded_and_closed_worm))
        all_contours = self.BinaryImageHandler.get_contours(self.thresholded_and_closed_worm)
        all_contour_areas = np.array([cv2.contourArea(contour) for contour in all_contours])
        potential_inner_contours_is = np.argwhere(np.logical_and(all_contour_areas<(outer_contour_area/2),  all_contour_areas>self.inner_loop_area_min)).flatten()
        if potential_inner_contours_is.size == 0:
            return None
        #potential_inner_contours = all_contours[potential_inner_contours_is]
        potential_inner_contours = [all_contours[i] for i in potential_inner_contours_is]
        potential_inner_contours_centroids = np.concatenate([self.BinaryImageHandler.get_centroid_from_contour(inner_contour)[None,:] for inner_contour in potential_inner_contours], axis =0) ##Check area_thres# it should not be in outer contours either.. 
        inner_contour_dist_from_outer_contour = np.linalg.norm(potential_inner_contours_centroids-outer_contour_centroid, axis = 1) 
        inner_contour_i = np.argmin(inner_contour_dist_from_outer_contour) 
        inner_worm_contour = potential_inner_contours[inner_contour_i] #potential_inner_contours[sorted_contours_i]

        return inner_worm_contour

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
        if not self.good_coords:
            return self.good_coords, np.array([]) #np.array(np.zeros(self.n_midline_pts, 2))
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
