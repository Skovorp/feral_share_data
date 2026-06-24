
import numpy as np
import cv2

class BinaryImageHandler():
    def __init__(self):
        pass

    def threshold_and_close_img(self, img, threshold, closing_kernel):

        thresholded_and_closed_img = (img<threshold).astype(np.uint8)
        thresholded_and_closed_img = cv2.morphologyEx(thresholded_and_closed_img, cv2.MORPH_CLOSE, closing_kernel)
        return thresholded_and_closed_img

    def get_contours_sorted_by_n_pts(self,binary_img):
        outer_contours, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        #outer_contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        sorted_contours = sorted(outer_contours, key = lambda x: x.shape[0], reverse = True)
        sorted_contours = [contour[:,0,:] for contour in sorted_contours]
        return sorted_contours
    # def get_outer_contours(self,binary_img):
    #     #outer_contours, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #     outer_contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #     #sorted_contours = sorted(outer_contours, key = lambda x: x.shape[0], reverse = True)
    #     outer_contours = [contour[:,0,:] for contour in outer_contours]
    #     return outer_contours
    def get_outer_contours(self,binary_img):
        #contours, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        outer_contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #sorted_contours = sorted(outer_contours, key = lambda x: x.shape[0], reverse = True)
        contours = [contour[:,0,:] for contour in outer_contours]
        return contours
    def get_contours(self,binary_img):
        contours, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        #outer_contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #sorted_contours = sorted(outer_contours, key = lambda x: x.shape[0], reverse = True)
        contours = [contour[:,0,:] for contour in contours]
        return contours
    def get_largest_contour_from_binary_img(self, binary_img):
        sorted_contours = self.get_contours_sorted_by_n_pts(binary_img)
        return sorted_contours[0]

    def get_centroid_from_contour(self, contour):
        moments = cv2.moments(contour)
        if moments['m00'] != 0.0:
            cx = moments['m10']/moments['m00']
            cy = moments['m01']/moments['m00']
            centroid = np.asarray([cx,cy])
        else:
            centroid = np.asarray([-1,-1])
        return centroid

    def get_centroid_from_binary_img(self, binary_img):
        moments = cv2.moments(binary_img)
        if moments['m00'] != 0.0:
            cx = moments['m10']/moments['m00']
            cy = moments['m01']/moments['m00']
            centroid = np.asarray([cx,cy])
        else:
            centroid = np.asarray([-1,-1])
        return centroid

    def draw_img_mask_from_contours(self, img_shape, contours):

        img_mask = np.zeros(img_shape).astype('uint8')
        cv2.drawContours(img_mask, list(contours), -1, (255,255,255), -1)

        return img_mask