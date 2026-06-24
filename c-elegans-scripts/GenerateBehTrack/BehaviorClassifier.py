#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 14:50:57 2022

@author: friederikebuck
"""
import copy

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

from VectorandImageHandlers.NPCurveHandler import NPCurveHandler
from VectorandImageHandlers.DebugImgHandler import DebugImgHandler
from FileHandlers.YamlHandler import YamlHandler

class BehaviorClassifier():
    def __init__(self,beh_debug_imgs_dir, params_yaml):

        self.params_yaml = params_yaml
        yaml_handler = YamlHandler()
        param_dict = yaml_handler.get_dictionary_from_yaml(params_yaml)["beh_class_params"]

        self.NPCurveHandler = NPCurveHandler()
        self.DebugImgHandler = DebugImgHandler(params_yaml)

        self.dframes = param_dict["dframes"]
        self.fps = param_dict["fps"]
        self.dt = self.dframes/self.fps

        self.speed_thres = param_dict["speed_thres"]
        self.high_speed_bending_thres = param_dict["high_speed_bending_thres"]
        self.low_speed_bending_thres = param_dict["low_speed_bending_thres"]
        self.pause_time_thres_s = param_dict["pause_time_thres_s"]
        
        self.scale_factor = param_dict["scale_factor"]
        self.arrow_thickness = param_dict["arrow_thickness"]

        self.is_turning = None
        self.speed = None
        self.velocity = None
        self.is_pausing = None
        self.is_reversing = None
        self.is_forwards = None

        self.vid_handler = None

        self.fontColor = [0,0,0]


        self.thickness = 1



        self.beh_debug_imgs_dir = ""

        self.beh_class = []
        
    def interpolate_feature(self, frame_times_csv, feature_to_interpolate, interpolation_fps):
        frame_times = pd.read_csv(frame_times_csv, sep=',', usecols=['frame_time']).to_numpy().flatten()
        midline_frame_times =  frame_times[self.start_frame:self.start_frame+self.n_frames]
        evenly_spaced_timepoints = self.NPCurveHandler.get_evenly_spaced_timepoints_along_array(frame_times, interpolation_fps)
        interpolated_feature = self.NPCurveHandler.interpolate_data_at_timepoints(feature_to_interpolate, evenly_spaced_timepoints, midline_frame_times)
        return evenly_spaced_timepoints, interpolated_feature


    def init_from_WormTrack_obj(self, WormTrack_obj, init_inverted = True, interpolated = False, frame_times_csv ="", interpolation_fps = -1):


        self.is_looping = WormTrack_obj.is_looping.astype('int')
        self.start_frame = WormTrack_obj.start_frame

        self.box_buffer_starts = WormTrack_obj.box_buffer_starts[:,[1,0]]

        self.n_frames = WormTrack_obj.n_frames
        self.start_frame = WormTrack_obj.start_frame
        self.vid_handler = WormTrack_obj.vid_handler
        frame_width = self.vid_handler.vid_width
        frame_height = self.vid_handler.vid_height
        
        self.img_half_size_coords = np.asarray([frame_height/2, frame_width/2])

        self.vid_handler.set_frame_to_read(self.start_frame)
        
        self.beh_class = [-1 for i in range(self.n_frames)]
        self.is_turning = np.ones(self.n_frames)*-1
        self.is_pausing = np.ones(self.n_frames)*-1
        self.is_forwards = np.ones(self.n_frames)*-1
        self.is_reversing = np.ones(self.n_frames)*-1
        self.speed = np.zeros(self.n_frames)
        self.velocity = np.zeros((self.n_frames, 2))
        #if init_inverted:
        self.is_inverted = np.zeros(self.n_frames)


        self.is_turning[:self.dframes] = np.nan
        self.is_pausing[:self.dframes] = np.nan
        self.is_forwards[:self.dframes] = np.nan
        self.is_reversing[:self.dframes] = np.nan
        self.speed[:self.dframes] = np.nan
        self.velocity[:self.dframes,:] = np.nan

        self.get_midline_supplemental_angle = WormTrack_obj.get_midline_supplemental_angles
        self.effective_bending_angles = WormTrack_obj.effective_bending_angles
        self.local_midline_coords = WormTrack_obj.oriented_midlines
        self.bad_frame = WormTrack_obj.bad_midline_frames

        self.local_head_coords = self.local_midline_coords[:,0,:]
        self.global_head_coords = self.local_head_coords + self.box_buffer_starts

        self.local_tail_coords = self.local_midline_coords[:,-1,:]
        self.global_tail_coords = self.local_tail_coords + self.box_buffer_starts

        self.local_centroids = WormTrack_obj.bbox_img_centroids
        self.global_centroids = self.local_centroids + self.box_buffer_starts
        
        self.global_centroids_to_use = self.global_centroids
        self.local_midline_coords_to_use = self.local_midline_coords
        
        # update_is_turning_track(oriented_midlines_track, is_looping_track, min_duration_thres = 3)
        
        # if interpolated:
        #     if interpolation_fps == -1:
        #         interpolation_fps = self.fps
        #     self.global_centroids_interpolated = self.interpolate_feature(frame_times_csv, self.global_centroids, interpolation_fps)
        #     self.local_midline_coords_interpolated = self.interpolate_feature(frame_times_csv, self.local_midline_coords,interpolation_fps)
        #     self.global_centroids_to_use = self.global_centroids_interpolated 
        #     self.local_midline_coords_to_use = self.local_midline_coords_interpolated
            
        


    def align_img_coords_to_stage_coords(self, img_coords,  stage_coords_cam):
        #recievign coords from cropped rescaled img and want to go back to coords cam uses
        #stage_coords is already in terms of the coord sytemt eh cam uses (img_coords is not)

        img_coords_cam = img_coords/self.scale_factor
        return img_coords_cam + stage_coords_cam - self.img_half_size_coords

    def get_midline_supplemental_angles(self, midline):
        midline_supp_angles = self.NPCurveHandler.get_signed_supplementary_angles_between_coords1(midline)
        midline_supp_angles = self.NPCurveHandler.convert_radians_to_degrees(midline_supp_angles)
        return midline_supp_angles

    def get_effective_bending_angle(self, midline):
        midline_supp_angles = self.get_midline_supplemental_angles(midline)

        window_size_thirds= round(midline.shape[0]/3)
        sliding_sum_bending_angle_thirds = np.convolve(midline_supp_angles,np.ones(window_size_thirds,dtype=int),'valid')
        max_bending_angle_thirds = max(np.abs(sliding_sum_bending_angle_thirds))

        window_size_halfs= round(midline.shape[0]/2)
        sliding_sum_bending_angle_halfs = np.convolve(midline_supp_angles,np.ones(window_size_halfs,dtype=int),'valid')
        max_bending_angle_halfs = max(np.abs(sliding_sum_bending_angle_halfs))

        effective_bending_angle = min(max_bending_angle_thirds, max_bending_angle_halfs)

        return effective_bending_angle, max_bending_angle_thirds, max_bending_angle_halfs


    def categorize_behavior(self,speed, effective_bending_angle, is_looping):

        if abs(speed)>self.speed_thres:
            if speed > 0:# if going at a high speed differentiate between turn forwarda nd abackwards
                beh = "Fwd"
            else:
                beh = "Rev"

            if effective_bending_angle>self.high_speed_bending_thres:
                beh = "Turn"

        else:# if going at a low speed differentiate between turn and pause
            if effective_bending_angle>self.low_speed_bending_thres:
                beh = "Turn"
            else:
                beh = "Pause"

        #if is looping then classific ation is turn always
        if is_looping:
            beh = "Turn"

        return beh

    def get_row_with_closest_frame_i(self, frame_array, frame_i):
        frame_is = frame_array[:, 0]
        row_i = np.argmin(frame_is - frame_i)
        return row_i

    
    
    def get_is_turning(self):
        self.is_turning = 0
        is_turning_high_speed = np.logical_and(self.effective_bending_angles>self.high_speed_bending_thres, self.speed>self.speed_thres)
        is_turning_low_speed = np.logical_and(self.effective_bending_angles>self.low_speed_bending_thres, self.speed<= self.speed_thres)
        self.is_turning = np.logical_or(is_turning_high_speed, is_turning_low_speed)
        self.is_turning[self.is_looping] = 1 
        
    def get_is_pausing(self):
        self.pausing = 0
        if np.any(self.is_turning==-1):
            print("RUN get_is_turning() first!")
        low_velocity_non_turning_frames = np.logical_and(abs(self.speed)<=self.speed_thres, self.effective_bending_angles<=self.low_speed_bending_thres)
        
        pause_time_thres_frames = self.pause_time_thres_s*self.fps
        slow_velocity_start_frames, slow_velocity_end_frames  = self.NPCurveHandler.get_start_end_is_of_ones_in_binary_array(low_velocity_non_turning_frames)
        slow_velocity_frames_durations = self.NPCurveHandler.get_durations_of_ones_in_binary_array(slow_velocity_start_frames, slow_velocity_end_frames)
        pause_start_is = np.argwhere(slow_velocity_frames_durations>pause_time_thres_frames)
        pause_start_frames = slow_velocity_start_frames[pause_start_is]
        pause_end_frames = slow_velocity_end_frames[pause_start_is]
        n_frames = self.is_forwards.shape[0]
        self.is_pausing= self.NPCurveHandler.get_binary_matrix_form_start_end_is(pause_start_frames, pause_end_frames, n_frames)
        
        self.is_pausing[self.is_turning] = 0
    
    def get_is_forwards(self):
        if np.any(self.is_turning==-1):
            print("RUN get_is_turning() first!")
        if np.any(self.pausing==-1):
            print("RUN get_is_pausing() first!")
        self.is_forwards = np.zeros(self.n_frames)
        self.is_forwards[self.speed>=0] = 1
        self.is_forwards[self.is_turning] = 0
        self.is_forwards[self.is_pausing] = 0
        
    def get_is_reversing(self):
        if np.any(self.is_turning==-1):
            print("RUN get_is_turning() first!")
        if np.any(self.pausing==-1):
            print("RUN get_is_pausing() first!")
        self.is_reversing = np.zeros(self.n_frames)
        self.is_reversing[self.speed<0] = 1
        self.is_reversing[self.is_turning] = 0
        self.is_reversing[self.is_pausing] = 0
        
        
        
        
        
    def get_behavior_classification(self, save_debug_images= True ):
        #take in WormTrack_obj -- have centroid; have head coord tail coords, oriented side coords
    #have is looping
    #would like to have chunk start and stop times as well -- maye only deal with the chunk times
    #get behavior based on these...
        #how to classify behaviro near turn -- not 1000% sure but can try

        ###first change local centroid coords to gloabl cnetroid coords
        ###next take velcotiy by getting dpos and dt #maybe shoudl limit this to chunks in between turing?..nah its fine i think

        #self.incorporate_frame_time()
        self.velocity = self.calculate_velocity()
        self.speed = self.calculate_speed()

        self.vid_handler.set_frame_to_read(self.start_frame+self.dframes)
        init_img = self.vid_handler.imgGrab()
        self.DebugImgHandler = DebugImgHandler(self.params_yaml, init_img = init_img)
        self.vid_handler.set_frame_to_read(self.start_frame+self.dframes)


        for frame_i in range(self.dframes, self.n_frames):
            debug_img = self.vid_handler.imgGrab()

            if self.bad_frame[frame_i]:
                beh_class = "bad frame"
                effective_bending_angle = -1
                self.save_behavior_class_debug_imgs(debug_img, frame_i, beh_class, effective_bending_angle, 0, 0)

                continue

            midline = self.local_midline_coords[frame_i, :,:]
            effective_bending_angle, max_bending_angle_thirds, max_bending_angle_halfs  = self.get_effective_bending_angle(midline)


            beh_class = self.categorize_behavior(self.speed[frame_i], effective_bending_angle, self.is_looping[frame_i])

            self.update_behavior_class_attributes(frame_i, beh_class)

            if save_debug_images:
                self.save_behavior_class_debug_imgs(debug_img, frame_i, beh_class, effective_bending_angle, max_bending_angle_thirds, max_bending_angle_halfs)
        #print("is turning frames", np.argwhere(self.is_turning))
        #print("is pausing frames", np.argwhere(self.is_pausing).flatten())
        
        #self.update_is_pausing()
        #########################
        ##newwer version 
        # self.velocity = self.calculate_velocity()
        # self.speed = self.calculate_speed()
        
        # #NOTE:ORDER MATTERS!!!!!!
        # self.get_is_turning()
        # self.get_is_pausing()
        # self.get_is_forwards()
        # self.get_is_reversing()
        
    def save_all_beh_class_debug_imgs(self, save_debug_images = True):
        if self.beh_class[-1] ==-1:
            self.update_beh_class_list()
        

        self.vid_handler.set_frame_to_read(self.start_frame+self.dframes)
        init_img = self.vid_handler.imgGrab()
        self.DebugImgHandler = DebugImgHandler(self.params_yaml, init_img = init_img)
        self.vid_handler.set_frame_to_read(self.start_frame+self.dframes)


        for frame_i in range(self.dframes, self.n_frames):
            debug_img = self.vid_handler.imgGrab()
            

            if self.bad_frame[frame_i]:
                beh_class = "bad frame"
                effective_bending_angle = -1
                self.save_behavior_class_debug_imgs(debug_img, frame_i, beh_class, effective_bending_angle, 0, 0)

                continue
            beh_class = self.beh_class[frame_i]
            # max_bending_angle_thirds = self.max_bending_angle_thirds[frame_i]
            # max_bending_angle_halfs =  self.max_bending_angle_halfs[frame_i]

            # self.save_behavior_class_debug_imgs(debug_img, frame_i, beh_class, effective_bending_angle, max_bending_angle_thirds, max_bending_angle_halfs)
            self.save_behavior_class_debug_imgs(debug_img, frame_i, beh_class, effective_bending_angle, 0, 0)


            self.update_behavior_class_attributes(frame_i, beh_class)

            if save_debug_images:
                self.save_behavior_class_debug_imgs(debug_img, frame_i, beh_class, effective_bending_angle, max_bending_angle_thirds, max_bending_angle_halfs)
        #print("is turning frames", np.argwhere(self.is_turning))
        # print("is pausing frames", np.argwhere(self.is_pausing).flatten())
        return self.speed

    
    # def update_is_pausing(self):
    #     self.low_speed = self.speed
    #     pause_time_thresh_frames = self.pause_time_thres_s/self.fps
        
        
    #     raise NotImplementedError

    def calculate_velocity(self):
        dposition = self.global_centroids_to_use[self.dframes:,:]-self.global_centroids_to_use[:-1*self.dframes,:] #four frames later minus for frames earlier
        velocity = dposition/self.dt
        #("velocity.shape", velocity.shape)
        velocity = np.concatenate([np.zeros((self.dframes, 2)), velocity], axis = 0)
        return velocity

    def calculate_head_tail_vector(self):

        head_tail_vector = self.global_head_coords-self.global_tail_coords
        return head_tail_vector

    def calculate_speed(self):
        #get magnitude of speed based on norm of velocity
        speed = np.linalg.norm(self.velocity, axis = 1)

        #get head tail vector
        head_tail_vector = self.calculate_head_tail_vector()

        #change sign of speed based on whehther worm is travellign in direction of  head or in diretion of tail
        for frame_i in range(self.dframes, self.n_frames):
            if self.NPCurveHandler.dot_product(head_tail_vector[frame_i,:].reshape((1,2)), self.velocity[frame_i,:].reshape((1,2)))<0:
                speed[frame_i] =-speed[frame_i]

        return speed
    
    def update_beh_class_list(self):
        for i in range(self.n_frames):
            if self.bad_frame[i]:
                self.beh_class = "bad frame"
            if self.is_turning[i]:
                self.beh_class[i] = "Turn"
            elif self.is_pausing[i]:
                self.beh_class[i] = "Pause"
            elif self.is_forwards[i]:
                self.beh_class[i] = "Fwd"
            elif self.is_forwards[i]:
                self.beh_class[i] = "Rev"
                
        

    def update_behavior_class_attributes(self,frame_i, beh_class):
        if beh_class == "Turn":
            self.is_turning[frame_i] = 1
        elif beh_class =="Pause":
            self.is_pausing[frame_i] = 1
        elif beh_class == "Fwd":
            self.is_forwards[frame_i] = 1
        elif beh_class =="Rev":
            self.is_reversing[frame_i] = 1

#save_behavior_class_debug_imgs(debug_img, frame_i, beh_class, effective_bending_angle, max_bending_angle_thirds, max_bending_angle_halfs)
    def save_behavior_class_debug_imgs(self,debug_img, frame_i, beh_class, effective_bending_angle,  max_bending_angle_thirds, max_bending_angle_halfs):


        self.dark_color = self.DebugImgHandler.darkest_color
        self.light_color = self.DebugImgHandler.lightest_color
        self.dark_gray = self.DebugImgHandler.get_mid_color(0.3)
        self.light_gray = self.DebugImgHandler.get_mid_color(0.7)

        text_color = self.dark_color
        arrow_color = self.dark_color



        if beh_class == "bad frame":
            debug_img= self.DebugImgHandler.put_text_on_img(debug_img,"bad img", text_position = (8,15), color = text_color)
            plt.imsave(self.beh_debug_imgs_dir+str(frame_i)+"_beh_debug.jpeg", debug_img, cmap = 'gray')

        #add behavior class
        debug_img= self.DebugImgHandler.put_text_on_img(debug_img,str(beh_class), text_position = (8,10), color = text_color)

        #add speed as text if not turning
        # if beh_class!="Turn":
        if frame_i ==2800:
            print(2800)
        debug_img= self.DebugImgHandler.put_text_on_img(debug_img,str(self.speed[frame_i]), text_position = (8, 115), color = text_color)
        debug_img= self.DebugImgHandler.put_text_on_img(debug_img,"inverted"+str(self.is_inverted[frame_i]), text_position = (8, 25), color = text_color)

        # #add effectvie bednign angle:
        # debug_img= self.DebugImgHandler.put_text_on_img(debug_img,str(effective_bending_angle), text_position = (8,75), color = black)
        # debug_img= self.DebugImgHandler.put_text_on_img(debug_img,str(max_bending_angle_thirds), text_position = (8,90), color = black)
        # debug_img= self.DebugImgHandler.put_text_on_img(debug_img,str(max_bending_angle_halfs), text_position = (8,105), color = black)



        #draw arrows on debug image
        tail_coords = tuple(self.local_tail_coords[frame_i,:])
        head_coords = tuple(self.local_head_coords[frame_i,:])
        velocity_endpt = tuple(self.local_tail_coords[frame_i,:]+self.velocity[frame_i-self.dframes,:])
        try:
            debug_img = self.DebugImgHandler.draw_arrow_on_debug_img(debug_img,tail_coords, head_coords, arrow_color, self.arrow_thickness)
            debug_img = self.DebugImgHandler.draw_arrow_on_debug_img(debug_img,tail_coords,velocity_endpt ,  arrow_color, self.arrow_thickness)
        except:
            print('problem drawing arrows')
            print(self.local_tail_coords[frame_i,:])
        self.DebugImgHandler.save_img(self.beh_debug_imgs_dir, str(frame_i)+"_beh_debug.jpeg", debug_img )

    # def add_dframe_buffer_tois_turning(self):

    #     turning_beh = self.is_turning[dframes:]
    #     pausing_beh = self.is_pausing[dframes:]





    def flip_head_tail_based_on_forward_rev_lengths(self, worm_track_obj, chunk_start_is, chunk_end_is, chunk_length_thres = 200):
        chunk_length_thres = 80#this would chakec that longer than
        rev_beh = np.zeros(self.n_frames)
        rev_beh_frames = np.argwhere(self.speed<0)
        rev_beh[rev_beh_frames] = 1
        rev_beh[np.argwhere(self.is_pausing)] = 0

        fwd_beh = np.zeros(self.n_frames)
        fwd_beh_frames = np.argwhere(self.speed>0)
        fwd_beh[fwd_beh_frames] = 1
        fwd_beh[np.argwhere(self.is_pausing)] = 0

        for i in range(chunk_start_is.shape[0]):
            chunk_start_i = chunk_start_is[i]
            if chunk_start_i ==2800:
                print("at chunk")
            chunk_end_i = chunk_end_is[i]

            chunk_length = chunk_end_i - chunk_start_i

            if chunk_length < chunk_length_thres:
                continue

            n_fwd_frames = np.argwhere(fwd_beh[chunk_start_i:chunk_end_i]).shape[0]
            n_rev_frames = np.argwhere(rev_beh[chunk_start_i:chunk_end_i]).shape[0]

            if n_rev_frames > n_fwd_frames:
                midlines_chunk = copy.deepcopy(self.local_midline_coords[chunk_start_i:chunk_end_i,:,:])
                inverted_midlines = np.flip(midlines_chunk, axis = 1)

                worm_track_obj.oriented_midlines[chunk_start_i:chunk_end_i,:,:] = inverted_midlines #note this inverted local midline coords as well..they are pointing to the same thign apparanely
                self.local_midline_coords[chunk_start_i:chunk_end_i,:,:] = inverted_midlines
                self.is_inverted[chunk_start_i:chunk_end_i] = 1
                continue
        return worm_track_obj