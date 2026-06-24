import numpy as np
import cv2
import glob
import shutil

from FileHandlers.ZipHandler import ZipHandler
from FileHandlers.OSHandler import OSHandler
#from FileHandlers.VideoHandlers.JpegVideoHandler import JpegVideoHandler#
#from FileHandlers.VideoHandlers.NPVideoHandler import NPVideoHandler
#from FileHandlers.VideoHandlers.AviVideoHandler import AviVideoHandler
#from VideoHandlers.ZippedVideoHandler import ZippedVideoHandler

class VideoHandler():
    def __init__(self, vid_folder_name, wkdir= None, is_compressed = False,  uncompressed_vid_type = 'jpeg'):
        self.vid_originally_compressed = is_compressed
        self.vid_folder_name = vid_folder_name
        self.zip_handler = ZipHandler()

        self.is_vid_compressed = is_compressed

        self.vid_handler = None
        self.wkdir= wkdir
    def imgGrab(self):
        pass

    def set_frame_to_read(self, frame_to_read):
        pass
    def set_vid_originally_compressed(self,vid_originally_compressed ):
        self.vid_originally_compressed = vid_originally_compressed

    def extract_vid_folder_if_compressed(self):
        if self.is_vid_compressed:

            self.compressed_folder_name = self.vid_folder_name
            len_file_suffix = len(".zip")
            if self.wkdir is None: 

                self.folder_to_extract_to = self.compressed_folder_name[:-1*len_file_suffix]
            else: 
                 self.folder_to_extract_to = self.wkdir
            self.zip_handler.extract_zipfile(self.compressed_folder_name, self.folder_to_extract_to)
            oshandler = OSHandler()
            uncompressed_vid_folder_names = oshandler.get_immediate_subdirectory_names_and_paths(self.folder_to_extract_to)
            print("uncompressed_vid_folder_names", uncompressed_vid_folder_names)
            self.uncompressed_vid_folder_name = uncompressed_vid_folder_names[0][1]
            self.is_vid_compressed = False
        else:
            self.uncompressed_vid_folder_name = self.vid_folder_name
        return self.uncompressed_vid_folder_name


    def delete_uncompressed_vid_folder(self):
        if self.vid_originally_compressed:
            print("in vid handler delete uncompressed vid folder " )
            print("removing vid: ",self.uncompressed_vid_folder_name)
            shutil.rmtree(self.uncompressed_vid_folder_name)