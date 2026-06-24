# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 07:45:21 2022

@author: fbuck
"""

import copy


import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

#import VectorandImageHandlers
#import GenerateBehTrack
#import FileHandlers


from FileHandlers.YamlHandler import YamlHandler
from FileHandlers.VideoHandlers.JpegVideoHandler import JpegVideoHandler

from GenerateBehTrack.WormShape.WormMorphology import WormMorphology
from GenerateBehTrack.WormShape.WormMidlineFeatures import WormMidlineFeatures
from GenerateBehTrack.TrackChunkHandler.MidlinesChunkObj import MidlinesChunkObj
from GenerateBehTrack.TrackChunkHandler.MidlinesChunkHandler import MidlinesChunkHandler

from VectorandImageHandlers.DebugImgHandler import DebugImgHandler
from VectorandImageHandlers.NPCurveHandler import NPCurveHandler

class WormTrack():
    def __init__(self, params_yaml):
        self.params_yaml = params_yaml
        yaml_handler = YamlHandler()
        self.NPCurveHandler = NPCurveHandler()
        param_dict = yaml_handler.get_dictionary_from_yaml(params_yaml)["worm_track_params"]
        self.save_debug_imgs = param_dict['save_debug_imgs']
        self.debug_img_folder= param_dict['debug_img_folder']
        self.n_midline_pts = param_dict['n_midline_pts']

        self.midline_coord_radius = param_dict['midline_coord_radius']
        self.extrema_coord_radius = param_dict['extrema_coord_radius']
        self.interpolation_fps = param_dict['interpolation_fps']
        self.start_frame = None

        self.n_frames = 0

        self.vid_handler  = None

        self.is_looping = np.array([])
        self.good_midline_frames = np.array([])
        self.bad_midline_frames = np.array([])

        self.unoriented_midlines = np.array([])
        self.oriented_midlines = np.array([])

        self.inner_worm_points = []
        self.bbox_img_centroids  = np.array([])

        self.bad_frame_count = 0

        self.bad_track = False
        
        

    def init_by_phil_tracker_output(self, vid_handler, start_frame, box_buffer_starts):

        self.start_frame = start_frame

        self.vid_handler = vid_handler
        self.n_frames = self.vid_handler.get_frame_count()
        self.vid_handler.set_frame_to_read(self.start_frame)

        #init now that know number of frames
        self.is_looping = np.zeros(self.n_frames)
        self.good_midline_frames = np.zeros(self.n_frames)
        self.bad_midline_frames = np.zeros(self.n_frames)
        self.unoriented_midlines = np.zeros((self.n_frames, self.n_midline_pts,2))
        self.oriented_midlines = np.zeros((self.n_frames, self.n_midline_pts,2))
        self.bbox_img_centroids = np.zeros((self.n_frames, 2))

        self.box_buffer_starts = box_buffer_starts
        
        self.get_midline_supplemental_angles = np.zeros((self.n_frames, self.n_midline_pts-1))
        self.effective_bending_angles = np.zeros(self.n_frames)

    def get_midlines(self):
        self.get_unoriented_midlines_and_features()
        if self.bad_frame_count>10:
            print("bad track skipping")
            self.bad_track = True
            return
        self.orient_midlines()



    def get_unoriented_midlines_and_features(self):
        '''
        gets unoriented midlines for each frame in worm track
        assumes the object has already been initalized for instance with init_by_phil_tracker_output()
        requires updated sorted_img_file_names


        for each frame updates:
        bbox_img_centroids: local position relative to bounding box

        is_looping: whether there is self intersection of the worm

        self.bad_frames: 0 if worm is self intersecting, if was able to resolve midline, and if midline if longer tahn midline length min

        self.unoriented_midlines: evenly spaced, ordered but unoriented coords of midline of worm if not self intersecting (if self intersecting then 0's) and not bad coords

        self.side_coords:evenly spaced, ordered but unoriented coords of randomly selected side of worm if not self intersecting

        self.inner_worm_points: coords of points inside the worm, minus a small border

        good_midline_frames
        '''

        #iteracte through each fraem in track and update bbox_img_centroids, is_looping, bad_frames, self.unoirented_midlines,  self.side_coords,self.inner_worm_points
        #as is specified above
        self.vid_handler.set_frame_to_read(self.start_frame)
        for frame_i in range(self.n_frames):
            bbox_img = self.vid_handler.imgGrab()
            if frame_i%500 ==0:
                 print(frame_i)
            if len(bbox_img.shape)!=2:
                #print('bad image')
                self.bad_frame_count+=1
                continue

            worm_morphology = WormMorphology(frame_i, bbox_img, self.params_yaml)

            #get centroid of worm at current frame
            centroid = worm_morphology.get_worm_centroid()
            self.bbox_img_centroids[frame_i, :] = centroid
            if np.any(centroid ==-1):
                worm_morphology.get_worm_centroid()

            #determine if worm is looping (self intersecting) at current frame
            islooping_bool = worm_morphology.detect_loop()
            if islooping_bool:
                self.is_looping[frame_i] = 1
                self.inner_worm_points.append(np.asarray([]))
                continue

            #if worm is not self intersecting get midline of the worm
            midline_success_bool, midline_coords = worm_morphology.get_midline()
            #print("midline_success_bool", midline_success_bool)
            if not midline_success_bool:
                self.bad_midline_frames[frame_i] = 1
                self.inner_worm_points.append(np.asarray([]))
                continue
            self.good_midline_frames[frame_i] = 1
            self.unoriented_midlines[frame_i,:,:] = midline_coords

            #get inner worm points
            inner_worm_points = worm_morphology.get_inner_worm_points()
            self.inner_worm_points.append(inner_worm_points)

    def orient_midlines(self):

        '''
        gets chunks of consecutive frames in which unoreinted  midline was well extracted

        for each chunk , orients previously unoriented midlines relative to head

        '''

        chunk_start_is, chunk_end_is = self.get_midline_chunk_indices()

        n_chunks = chunk_start_is.shape[0]



        #for each chunk, get midline coords, create mdiliens chunk obj, oriente mdilines relative to headand save
        for i in range(n_chunks):
            chunk_start_i = chunk_start_is[i]
            chunk_end_i = chunk_end_is[i]
            midlines_chunk = self.unoriented_midlines[chunk_start_i:chunk_end_i,:,:]

            inner_worm_points_chunk = self.inner_worm_points[chunk_start_i:chunk_end_i]
            midlines_chunk_obj = MidlinesChunkObj(self.start_frame,chunk_start_i, chunk_end_i, midlines_chunk,  inner_worm_points_chunk,self.bad_midline_frames)
            midlines_chunk_handler = MidlinesChunkHandler(midlines_chunk_obj, self.vid_handler, self.params_yaml)

            midlines_chunk_handler.orient_midlines_in_single_direction_relative_to_head()

            self.oriented_midlines[chunk_start_i:chunk_end_i,:,:] = midlines_chunk_handler.midlines_chunk

    def get_midline_chunk_indices(self):
        '''
        get frames in which consecutive chunk of good midlines or a "midline_chunk" starts and ends
        '''
        chunk_start_is, chunk_end_is = self.NPCurveHandler.get_start_end_is_of_ones_in_binary_array(self.good_midline_frames)
        # good_midline_frames_buffered = np.concatenate([np.zeros(1),self.good_midline_frames, np.zeros(1)])
        # chunk_start_is = np.argwhere(np.diff(good_midline_frames_buffered)==1).flatten()
        # chunk_end_is = np.argwhere(np.diff(good_midline_frames_buffered)==-1).flatten()

        return chunk_start_is, chunk_end_is


    def interpolate_feature(self, frame_times_csv, feature_to_interpolate):
        frame_times = pd.read_csv(frame_times_csv, sep=',', usecols=['frame_time']).to_numpy().flatten()
        midline_frame_times =  frame_times[self.start_frame:self.start_frame+self.n_frames]
        evenly_spaced_timepoints = self.np_curve_handler.get_evenly_spaced_timepoints_along_array(frame_times, self.interpolation_fps)
        interpolated_feature = self.np_curve_handler.interpolate_data_at_timepoints(feature_to_interpolate, evenly_spaced_timepoints, midline_frame_times)
        return evenly_spaced_timepoints, interpolated_feature

        
    def create_and_save_debug_imgs(self, debug_img_dir):

        self.vid_handler.set_frame_to_read(self.start_frame)
        init_img = self.vid_handler.imgGrab()
        debug_img_handler = DebugImgHandler(self.params_yaml, init_img= init_img)
        self.vid_handler.set_frame_to_read(self.start_frame)

        dark_color = debug_img_handler.darkest_color
        light_color = debug_img_handler.lightest_color
        dark_gray = debug_img_handler.get_mid_color(0.3)
        light_gray = debug_img_handler.get_mid_color(0.7)

        text_color = dark_color
        midline_coord_color = light_color
        head_color =  dark_gray
        tail_color = light_gray

        self.vid_handler.set_frame_to_read(self.start_frame)
        for frame in range(self.n_frames):
            img = self.vid_handler.imgGrab()

            midline = self.oriented_midlines[frame, :, : ]
            head_coords = self.oriented_midlines[frame, 0, : ]#.astype('uint8')
            tail_coords = self.oriented_midlines[frame, -1, : ]#.astype('uint8')

            is_looping_bool = self.is_looping[frame]

            if is_looping_bool:
                img = debug_img_handler.put_text_on_img(img, "looping", text_position = (5,5), color = text_color)

            if self.bad_midline_frames[frame]:
                img = debug_img_handler.put_text_on_img(img,  "bad frame", text_position = (5,5), color = text_color)

            if not is_looping_bool:
                for i in range(midline.shape[0]):
                    coord_in_midline= midline[i,:]

                    img = debug_img_handler.draw_coords_on_img(coord_in_midline, img, coord_radius=self.midline_coord_radius, coord_color = midline_coord_color, coord_thickness = -1)
                img = debug_img_handler.draw_coords_on_img(head_coords, img, coord_radius= self.extrema_coord_radius, coord_color = head_color, coord_thickness = -1)
                img = debug_img_handler.draw_coords_on_img(tail_coords, img, coord_radius= self.extrema_coord_radius, coord_color = tail_color, coord_thickness = -1)

            plt.imsave(debug_img_dir+str(frame)+"_debug.jpeg", img, cmap = 'gray')
            
    def get_midline_supplemental_angle_track(self):
        if np.sum(self.oriented_midlines) ==0: 
            raise "get oritnted midlines first"
        self.get_midline_supplemental_angles = np.zeros((self.n_frames, self.n_midline_pts-1))
        self.effective_bending_angles = np.zeros(self.n_frames)
        
        worm_midline_features = WormMidlineFeatures()
    
        for frame_i in range(self.n_frames):
                

                if self.bad_frame[frame_i]:
                   

                    continue
                
                midline = self.oriented_midlines[frame_i, :,:]
                
                midline_supplemental_angle = worm_midline_features.get_midline_supplemental_angles(midline)

                effective_bending_angle, max_bending_angle_thirds, max_bending_angle_halfs  = worm_midline_features.get_effective_bending_angle(midline, midline_supp_angles = midline_supplemental_angle)
            
                self.get_midline_supplemental_angles[frame_i, :] = midline_supplemental_angle
                self.effective_bending_angles[frame_i] = effective_bending_angle
                
    def get_eigenworm():
        pass
    def project_eigenworms():
        pass