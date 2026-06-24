import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image

from FileHandlers.YamlHandler import YamlHandler

class DebugImgHandler():
    def __init__(self, params_yaml, init_img = []):#, save_img_dir):
        #self.save_img_dir = save_img_dir
        yaml_handler = YamlHandler()
        param_dict = yaml_handler.get_dictionary_from_yaml(params_yaml)["debug_image_params"]

        self.fontScale = param_dict["fontScale"]
        font_str = param_dict["font"]
        if font_str == "hershey_simplex":
            self.font = cv2.FONT_HERSHEY_SIMPLEX
        else:
            raise TypeError("font not implemented")

        self.unsigned_int_types = set(['uint8', 'uint16', 'uint32'])
        self.signed_int_types = set(['int8', 'int16', 'int32'])
        #self.coord_color = [-9999, -9999,-9999]

        if np.asarray(init_img).size ==0:
            self.darkest_color = [0,0,0]
            self.lightest_color = [255,255,255]
        else:
            self.darkest_color = self.get_darkest_color(init_img)
            self.lightest_color = self.get_lightest_color(init_img)

    def get_darkest_color(self, img):
        if str(img.dtype) in self.unsigned_int_types:
            darkest_color = [0,0,0]
        elif str(img.dtype) in self.signed_int_types:
            darkest_color_val = int(min(np.unique(img)))
            darkest_color = [darkest_color_val,darkest_color_val,darkest_color_val]
        else:
            raise TypeError("img not of specified np type")

        return darkest_color

    def get_lightest_color(self, img):
        if str(img.dtype) in self.unsigned_int_types:
            lightest_color = [255,255,255]
        elif str(img.dtype) in self.signed_int_types:
            lightest_color_val = int(max(np.unique(img)))
            lightest_color = [lightest_color_val,lightest_color_val,lightest_color_val]
        else:
            raise TypeError("img not of specified np type")

        return lightest_color

    def get_mid_color(self,frac):
        color_range = self.lightest_color[0] - self.darkest_color[0]
        mid_color_val =  int(self.darkest_color[0]) + color_range*frac
        mid_color = [mid_color_val, mid_color_val, mid_color_val]
        return mid_color

    def draw_coords_on_img(self, coords, img, coord_radius=3, coord_color = [], coord_thickness = -1):

        #img = img.astype('uint8')

        if len(coord_color) == 0:
            coord_color = self.get_darkest_color(img)


        if len(coords.shape) == 1:

            cv2.circle(
                    img,
                    (int(coords[0]), int(coords[1])),
                    radius=coord_radius,
                    color=coord_color,
                    thickness=coord_thickness,
                )

        elif len(coords.shape) == 2:
            for i in range(coords.shape[0]):
                coord = coords[i,:]

                cv2.circle(
                    img,
                    (int(coord[0]), int(coord[1])),
                    radius=coord_radius,
                    color=coord_color,
                    thickness=coord_thickness,
                )
        else:
            print("please reshape coords")
        return img

    def put_text_on_img(self, img, text, text_position = (5,5), color = (255,255,255)):
        #img = img.astype('uint8')

        img = cv2.putText(img, text, text_position, cv2.FONT_HERSHEY_SIMPLEX,self.fontScale, color, 1,cv2.LINE_AA)
        return img

    def draw_arrow_on_debug_img(self, img,  start_point, end_point, arrow_color, arrow_thickness):
        #img = img.astype('uint8')
        start_point = (int(start_point[0]), int(start_point[1]))
        end_point = (int(end_point[0]), int(end_point[1]))
        img = cv2.arrowedLine(img, start_point, end_point, arrow_color,arrow_thickness)
        return img
    def save_img(self,save_img_dir,  file_name, img, cmap_str ='gray'):
        plt.imsave(save_img_dir+str(file_name), img, cmap = cmap_str)
        #cv2.imwrite(save_img_dir+str(file_name), img)
        #img = Image.fromarray(img).convert('RGB')
        #img.save(save_img_dir+str(file_name))

        # def draw_coords_on_debug_img(self, coords, debug_img, coord_radius=1, coord_color = [255, 255, 255], coord_thickness = -1):

        # if len(coords.shape) ==1:

        #     cv2.circle(
        #             debug_img,
        #             (int(coords[0]), int(coords[1])),
        #             radius=coord_radius,
        #             color=coord_color,
        #             thickness=coord_thickness,
        #         )

        # elif len(coords.shape) == 2:
        #     for i in range(coords.shape[0]):
        #         coord = coords[i,:]
        #         cv2.circle(
        #             debug_img,
        #             (int(coord[0]), int(coord[1])),
        #             radius=coord_radius,
        #             color=coord_color,
        #             thickness=coord_thickness,
        #         )
        # else:
        #     print("please reshape coords")
        # return debug_img