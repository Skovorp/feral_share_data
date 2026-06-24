import matplotlib.pyplot as plt
import numpy as np
import cv2

from GenerateBehTrack.WormShape.Worm_Midline import Worm_Midline

from VectorandImageHandlers.NPCurveHandler import NPCurveHandler
from VectorandImageHandlers.DebugImgHandler import DebugImgHandler
from VectorandImageHandlers.BinaryImageHandler import BinaryImageHandler

from FileHandlers.YamlHandler import YamlHandler

class WormMidlineFeatures():
    def __init__(self, frame_i, bbox_img, params_yaml):
        pass
    
    def get_midline_supplemental_angles(self, midline):
        midline_supp_angles = self.NPCurveHandler.get_signed_supplementary_angles_between_coords1(midline)
        midline_supp_angles = self.NPCurveHandler.convert_radians_to_degrees(midline_supp_angles)
        return midline_supp_angles

    def get_effective_bending_angle(self, midline, midline_supp_angles = ""):
        if midline_supp_angles=="":
            midline_supp_angles = self.get_midline_supplemental_angles(midline)

        window_size_thirds= round(midline.shape[0]/3)
        sliding_sum_bending_angle_thirds = np.convolve(midline_supp_angles,np.ones(window_size_thirds,dtype=int),'valid')
        max_bending_angle_thirds = max(np.abs(sliding_sum_bending_angle_thirds))

        window_size_halfs= round(midline.shape[0]/2)
        sliding_sum_bending_angle_halfs = np.convolve(midline_supp_angles,np.ones(window_size_halfs,dtype=int),'valid')
        max_bending_angle_halfs = max(np.abs(sliding_sum_bending_angle_halfs))

        effective_bending_angle = min(max_bending_angle_thirds, max_bending_angle_halfs)

        return effective_bending_angle, max_bending_angle_thirds, max_bending_angle_halfs
