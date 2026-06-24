import numpy as np
import matplotlib.pyplot as plt
import cv2

from FileHandlers.VideoHandlers.VideoHandler import VideoHandler

class AviVideoHandler(VideoHandler):
    def __init__(self,video_file_name, is_compressed = False):

        super().__init__(video_file_name,  is_compressed = is_compressed)
        self.extract_vid_folder_if_compressed()

        self.video_file_name = self.uncompressed_vid_folder_name

        self.video_file_name = video_file_name
        self.vid = cv2.VideoCapture(self.video_file_name)
        self.frame_count = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))
        self.vid_height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.vid_width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))

    def imgGrab(self):#make video handler class for imgGrab

        ret,img = self.vid.read()
        if ret:
            return img[:,:,0]
        else:
            print('no image grabbed')
            return np.asarray([0,0])

    def set_frame_to_read(self, frame_to_read):
        self.vid.set(cv2.CAP_PROP_POS_FRAMES,frame_to_read)