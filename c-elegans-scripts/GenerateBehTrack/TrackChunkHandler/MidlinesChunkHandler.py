import numpy as np
import os
import sys
#from WormMorphology import WormMorphology
#from DebugImgHandler import DebugImgHandler
from GenerateBehTrack.TrackChunkHandler.WormSectionBrightnessHandlers.AdjacentWormSectionBrightness import AdjacentWormSectionMedianBrightness
from GenerateBehTrack.TrackChunkHandler.WormSectionBrightnessHandlers.WormExtremaSectionMeanBrightness import WormExtremaSectionMeanBrightness
#SCRIPT_DIR = os.path.dirname(os.path.abspath(os.path.join("..", "..","WormSectionBrightnessHandlers")))
#sys.path.append(os.path.dirname(SCRIPT_DIR))
#SCRIPT_DIR = os.path.dirname(os.path.abspath(os.path.join("..","VectorandImageHandlers")))
#sys.path.append(os.path.dirname(SCRIPT_DIR))
#from WormSectionBrightnessHandlers.AdjacentWormSectionBrightness import AdjacentWormSectionMedianBrightness
#from WormSectionBrightnessHandlers.WormExtremaSectionMeanBrightness import WormExtremaSectionMeanBrightness

from FileHandlers.YamlHandler import YamlHandler

import os

class MidlinesChunkHandler(): #made of a bunch of midlines

    def __init__(self, midlines_chunk_obj, vid_handler, params_yaml, save_section_imgs = False, save_img_dir = ""):

        #self.DebugImgHandler = DebugImgHandler()

        self.n_extrema = 2

        self.chunk_start_i = midlines_chunk_obj.chunk_start_i
        self.chunk_end_i =  midlines_chunk_obj.chunk_end_i
        self.start_frame = midlines_chunk_obj.start_frame
        self.bad_frames = midlines_chunk_obj.bad_frames
        self.midlines_chunk = midlines_chunk_obj.midlines_chunk
        self.inner_worm_points_chunk = midlines_chunk_obj.inner_worm_points_chunk

        self.params_yaml = params_yaml
        yaml_handler = YamlHandler()
        param_dict = yaml_handler.get_dictionary_from_yaml(params_yaml)["worm_chunk_params"]

        self.vid_handler = vid_handler
        self.vid_handler.set_frame_to_read(self.chunk_start_i+self.start_frame)

        self.n_frames_in_chunk = self.chunk_end_i-self.chunk_start_i

        self.midlines_oriented_in_single_direction = False
        self.midlines_oriented_in_single_direction_relative_to_head = False

        self.save_img_dir =  save_img_dir
        self.save_section_imgs =  save_section_imgs

        self.invert_bool = False

    def orient_midlines_in_single_direction(self):
        '''
        orient all midlines specified by self.midlines_chunk same direction (may not be directoin of the head)
        based on distance between adjacent midlines in current vs inverted direction

        outputs:
        uncentered_oriented_midlines_chunk: midlines in  self.midlines_chunk now all oritneted in smae direction
        inverted_bools: bool array with value 1 if original midline in self.midline_chunk was inverted and 0 ow

        '''
        centered_midlines_chunk, midline_center_coords = self.recenter_midlines(self.midlines_chunk)
        inverted_midlines_chunk = np.flip(centered_midlines_chunk, axis = 1)

        oriented_midlines_chunk = np.zeros(self.midlines_chunk.shape)
        inverted_bools = np.zeros(self.n_frames_in_chunk)

        oriented_midlines_chunk[0,:,:] = centered_midlines_chunk[0,:,:]
        for i in range(1, self.n_frames_in_chunk):

            org_midline = centered_midlines_chunk[i,:,:]
            inverted_midline = inverted_midlines_chunk[i,:,:]
            #prev_midline = oriented_midlines_chunk[i-1,:,:]
            #oriented_midlines_chunk[i,:,:], inverted_bools[i], distance =self.compare_adjacent_midline_distances(prev_midline,org_midline, inverted_midline)
            n = min(i-1, 5)
            prev_i = i-n
            prev_n_midlines = oriented_midlines_chunk[prev_i:i,:,:]
            oriented_midlines_chunk[i,:,:], inverted_bools[i], distance = self.compare_prev_n_midline_distances(prev_n_midlines, org_midline, inverted_midline)
            #distances[i,:] = distance

        uncentered_oriented_midlines_chunk = midline_center_coords[:,None,:]+oriented_midlines_chunk
        return uncentered_oriented_midlines_chunk, inverted_bools#, distances



    def recenter_midlines(self, midlines):
        '''recenter midlines such that center point of midline is at coord (0,0)

        inputs:
        midlines: np array shape (n_frames_in_chunk,n_midline_pts,2) midlines to recenter

        outputs:
        recentered midlines
        '''
        n_midline_coords = midlines.shape[1]#check

        midline_center_coord_i = np.floor(n_midline_coords/2).astype('uint8')
        midline_center_coords = midlines[:,midline_center_coord_i,:]
        recentered_midlines_chunk = midlines-midline_center_coords[:,None,:]

        return recentered_midlines_chunk, midline_center_coords

    def compare_prev_n_midline_distances(self, prev_n_midlines, org_midline, inverted_midline):

        distances_org_midline = np.sum(np.linalg.norm(prev_n_midlines - org_midline[None, :,:], axis =2), axis = 1)
        distances_inverted_midline = np.sum(np.linalg.norm(prev_n_midlines - inverted_midline[None, :,:], axis =2), axis = 1)

        total_distance_org = np.sum(distances_org_midline)
        total_distance_inverted = np.sum(distances_inverted_midline)

        distances = np.asarray([total_distance_org,total_distance_inverted])
        min_distance_i = np.argmin(distances)

        midlines = np.asarray([org_midline, inverted_midline])
        distance_compared_midline = midlines[min_distance_i]

        inverted_bool = min_distance_i

        return distance_compared_midline, inverted_bool , distances
    def compare_adjacent_midline_distances(self, prev_midline, org_midline, inverted_midline):
        '''compare sum distances between:
          coords at same indices of coords of midline at prev frame vs midline at current frame
          vs
          coords at same indices of coords of midline at prev frame vs inverted midline at current frame
        and track midline with shorter distance

        inputs:

        outputs:

        '''
        midlines = np.asarray([org_midline, inverted_midline])
        distances = np.sum(np.linalg.norm(midlines - prev_midline, axis =2), axis = 1)
        min_distance_i = np.argmin(distances)

        distance_compared_midline = midlines[min_distance_i, :,:]

        inverted_bool = min_distance_i

        return distance_compared_midline, inverted_bool , distances
    def orient_midlines_in_single_direction_relative_to_head(self):
        '''orient all midlines specified by self.midlines_chunk same direction relative to head (so that index 0 of midline is head coords and end index is the tail coords

        updates self.midlines_chunk directly to be oriented and updates the bools:
        self.midlines_oriented_in_single_direction and  self.midlines_oriented_in_single_direction_relative_to_head
        '''
        uncentered_oriented_midlines_chunk, _  = self.orient_midlines_in_single_direction()
        #print("oriented in singel direction")
        self.midlines_chunk = uncentered_oriented_midlines_chunk
        self.midlines_oriented_in_single_direction = True

        self.invert_bool, sum_mean_brightness = self.get_HT_brightness_vote_chunk()
        invert_bool_2, portion_invert_votes = self.get_HT_brightness_median_diff_vote()

        if int(self.invert_bool) != int(invert_bool_2 ):
            print("int(self.invert_bool) != int(invert_bool_2 )")
            print("self.chunk_start_i", self.chunk_start_i)
            print("self.chunk_end_i", self.chunk_end_i)
        else:
            # print("int(self.invert_bool) == int(invert_bool_2 )")
            # print("get_HT_brightness_vote_chunk", self.invert_bool, sum_mean_brightness )
            # print("get_HT_brightness_diff_vote_chunk", invert_bool_2, portion_invert_votes  )
            # print("self.chunk_start_i", self.chunk_start_i)
            # print("self.chunk_end_i", self.chunk_end_i)
            pass
        if invert_bool_2:
        #if invert_bool_2:
            self.midlines_chunk = np.flip(self.midlines_chunk, axis = 1)
        self.midlines_oriented_in_single_direction_relative_to_head = True

    def get_HT_brightness_vote_chunk(self):

        if not self.midlines_oriented_in_single_direction:
            self.orient_midlines_in_single_direction()

        sum_mean_brightness = self.get_sum_mean_brightness_around_each_extrema_over_all_frames_in_chunk()

        head_i = np.argmax(sum_mean_brightness)

        invert_bool = head_i
        return invert_bool, sum_mean_brightness


    def get_sum_mean_brightness_around_each_extrema_over_all_frames_in_chunk(self):
        sum_mean_brightness = np.zeros(self.n_extrema)
        self.vid_handler.set_frame_to_read(self.chunk_start_i+self.start_frame)
        for i in range(self.n_frames_in_chunk):
            frame_i = self.chunk_start_i+i
            img = self.vid_handler.imgGrab()

            if self.bad_frames[frame_i]:
                continue
            midline_coords = self.midlines_chunk[i,:,:].astype('uint16')
            inner_worm_coords = self.inner_worm_points_chunk[i]
            worm_extrema_mean_brightness_handler = WormExtremaSectionMeanBrightness(inner_worm_coords, midline_coords, img, self.params_yaml)
            sum_mean_brightness+= worm_extrema_mean_brightness_handler.get_mean_pixel_brightness_around_both_extrema()

        return sum_mean_brightness

    def get_HT_brightness_median_diff_vote(self):
        if not self.midlines_oriented_in_single_direction:
            self.orient_midlines_in_single_direction()
        sum_invert_votes = self.get_HT_brightness_median_diff_vote_chunk()

        invert_chunk_bool, portion_invert_votes = self.invert_vote_bool_based_on_majority(sum_invert_votes)

        return invert_chunk_bool, portion_invert_votes

    def get_HT_brightness_median_diff_vote_chunk(self):

        #invert_votes = np.zeros(self.n_frames_in_chunk)
        invert_vote = 0
        self.vid_handler.set_frame_to_read(self.chunk_start_i+self.start_frame)
        #extrema = [side_evenly_spaced_pts[0,:], side_evenly_spaced_pts[-1,:]]
        for j in range(self.n_frames_in_chunk):
            frame = self.chunk_start_i+j
            img = self.vid_handler.imgGrab()
            if self.bad_frames[frame]:
                continue

            inner_worm_coords = self.inner_worm_points_chunk[j]
            midline = self.midlines_chunk[j,:,:].astype('uint16')
            invert_vote_by_section_brightness_diff_handler = AdjacentWormSectionMedianBrightness(img, inner_worm_coords, midline, self.params_yaml)
            #invert_votes[j] = invert_vote_by_section_brightness_diff_handler.compare_median_brightness_difference_of_adjacent_sections_around_extremas()
            invert_vote+=invert_vote_by_section_brightness_diff_handler.compare_median_brightness_difference_of_adjacent_sections_around_extremas()

        #um_invert_votes = sum(invert_votes)
        return invert_vote
        #return sum_invert_votes
    def invert_vote_bool_based_on_majority(self, n_invert_votes):
        invert_chunk_bool = False
        if n_invert_votes> self.n_frames_in_chunk/2:
            invert_chunk_bool = True
        portion_invert_votes = n_invert_votes/self.n_frames_in_chunk
        return invert_chunk_bool, portion_invert_votes