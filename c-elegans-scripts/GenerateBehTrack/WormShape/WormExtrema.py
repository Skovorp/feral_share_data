

import numpy as np
import cv2

from VectorandImageHandlers.NPCurveHandler import NPCurveHandler
from VectorandImageHandlers.BinaryImageHandler import BinaryImageHandler

class WormExtrema():
    def __init__(self):
        self.NPCurveHandler = NPCurveHandler()
        self.BinaryImageHandler = BinaryImageHandler()
    def get_extrema(self, method = "get_extrema_from_midline"):

        if method == "get_extrema_from_midline":
            try:
                extrema = self.get_extrema_from_worm_midline(self.worm_midline_coords)
                self.extrema = extrema
            except:
                print("getting extrema from midlne unscuccessful")

        elif method == "get_extrema_from_corners":
            self.get_extrema_from_corners()

        elif method == "get_extrema_from_contour_curvature":
            self.get_extrema_from_contour_curvature()

        else:
            print(method+" does not exist!")
        self.extrema = extrema
        return self.extrema

    def get_extrema_from_worm_midline(self, midline_coords):
        return [midline_coords[0,:], midline_coords[-1,:]]

    def get_extrema_from_corners(self):
        print("get_extrema_from_corners not implemented yet!")

    def get_extrema_from_contour_curvature(self):
        print("get_extrema_from_contour_curvature not implemented yet!")

    def get_worm_side_coords(self, extrema, outer_worm_contour, n_pts_to_sample):

        extrema_is = [self.NPCurveHandler.find_nearest_coordinate_pair_in_list(extrema[i], outer_worm_contour) for i in range(len(extrema))]
         #get evenly spaced side coords

        side1, side2 = self.NPCurveHandler.get_segment_coords_between_two_indices(extrema_is[0], extrema_is[1], outer_worm_contour)

        side1_even = self.NPCurveHandler.get_even_points_along_curve(side1, n_pts_to_sample= n_pts_to_sample)

        return side1_even

    def get_corners(self, img, closing_kernel, outer_worm_contour, k_blur = 3, blur_thres = 190, k_blur_mask = 5, cornerHarris_params = [5,3,0.05], cornerHarris_thres = 3e-5):

        corners_img = cv2.blur(img,(k_blur,k_blur))

        corners_img = (corners_img<blur_thres).astype(np.uint8)

        closing = cv2.morphologyEx(corners_img, cv2.MORPH_CLOSE, closing_kernel)

        outer_worm_contour = self.get_outer_contour(closing)
        worm_mask = self.get_worm_mask_from_outer_contour(closing.shape, outer_worm_contour)

        worm_mask_blur = cv2.blur(worm_mask,(k_blur_mask,k_blur_mask))

        blocksize, ksize, k = cornerHarris_params
        #dst = cv2.cornerHarris(worm_mask, blocksize, ksize, k)
        #cornerHarris_thres = 3e-12

        dst = cv2.cornerHarris(worm_mask_blur, blocksize, ksize, k)

        extrema_mat = np.zeros(dst.shape, dtype='uint8')

        extrema_mat[dst>cornerHarris_thres]=1#-- get two points

        extrema_contours, _ = cv2.findContours(extrema_mat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        extrema_contours = sorted(extrema_contours, key = lambda x: cv2.contourArea(x), reverse = True)[0:2]

        extrema_contour_centers = [self.BinaryImageHandler.get_centroid_from_contour(x[:,0,[1,0]]) for x in extrema_contours]

        return extrema_contour_centers