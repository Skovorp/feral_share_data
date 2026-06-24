import numpy as np
# import cv2
import glob
from pathlib import Path
import os

from FileHandlers.VideoHandlers.VideoHandler import VideoHandler

class NPVideoHandler(VideoHandler):
    def __init__(self,np_folder_name, is_compressed = False ):
        super().__init__(np_folder_name,  is_compressed = is_compressed)
        self.extract_vid_folder_if_compressed()

        self.np_folder_name = self.uncompressed_vid_folder_name

        self.np_folder_name = np_folder_name
        #self.file_name_start_i = len(self.np_folder_name)
        self.file_root = ".npy"
        #self.file_name_root_len = len(self.file_root)

        self.sorted_img_file_names = self.get_sorted_img_file_names()
        self.frame_count = len(self.sorted_img_file_names)

        init_img_file_name = self.sorted_img_file_names[0]
        init_img = self.load_frame(init_img_file_name)
        self.vid_height,  self.vid_width = init_img.shape


        self.start_frame = self.get_frame_num_from_img_file_name(init_img_file_name)
        self.frame_to_read = self.start_frame

    def get_sorted_img_file_names(self):

        imgs_file_names = glob.glob(os.path.join(self.np_folder_name, "*"+ self.file_root+"*"))

        sorted_img_file_names = sorted(imgs_file_names, key = lambda x: self.get_frame_num_from_img_file_name(x))
        return sorted_img_file_names

    def load_frame(self, frame_file_name):
        return np.load(frame_file_name)
    def imgGrab(self, frame_to_read = -1):#make video handler class for imgGrab
        if frame_to_read > -1:
            self.frame_to_read = frame_to_read

        try:
            sorted_img_file_names_i = self.convert_frame_to_read_to_sorted_img_file_names_index(self.frame_to_read)
            img_file_name = self.sorted_img_file_names[sorted_img_file_names_i]
            img = self.load_frame(img_file_name)
            self.frame_to_read = self.frame_to_read+1
            return img
        except:
            print('no image grabbed')
            self.frame_to_read = self.frame_to_read+1
            return np.asarray([0,0])
    #def grab_next_frame():
    def get_frame_num_from_img_file_name(self, img_file_name):
        return int(Path(img_file_name).stem)
        #return int(img_file_name[self.file_name_start_i:-1*self.file_name_root_len])

    def convert_frame_to_read_to_sorted_img_file_names_index(self, frame_num):
        return frame_num - self.start_frame

    def set_frame_to_read(self, frame_to_read):
        self.frame_to_read = frame_to_read

    def get_frame_count(self):
        return self.frame_count