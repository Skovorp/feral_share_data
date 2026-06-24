import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import scipy.spatial as spatial
import cv2
import itertools
from skimage.morphology import skeletonize

from VectorandImageHandlers.NPCurveHandler import NPCurveHandler

from FileHandlers.YamlHandler import YamlHandler

class Worm_Midline():
    def __init__(self, params_yaml):
        self.midline_coords = np.asarray([])
        self.good_coords = False

        self.NPCurveHandler = NPCurveHandler()

        self.params_yaml = params_yaml
        yaml_handler = YamlHandler()
        param_dict = yaml_handler.get_dictionary_from_yaml(params_yaml)["worm_midline_params"]


        self.n_midline_pts = param_dict['n_midline_pts']

        self.midline_length_thres = param_dict['midline_length_thres']

    def get_midline_from_worm_mask(self, bin_worm_mask):
        skel_coords = self.skeletonize(bin_worm_mask)

        skel_coords = skel_coords[:,[1,0]]

        pts,G = self.init_Connectivity_graph(skel_coords)

        degreesD = G.degree(pts)
        degrees = np.asarray([degreesD[n] for n in pts])

        skel_extrema_is = np.argwhere(degrees == 1).flatten()
        skel_extrema_pts = skel_coords[skel_extrema_is,:]

        n_extrema = skel_extrema_is.shape[0]

        #skel_extrema_pts
        # if n_extrema ==1:
        #     self.good_coords = False
        #     self.midline_coords = skel_coords
        if n_extrema == 2: # if2 extrema (head and tail find shortest apth between )
            self.good_coords = True
            c = nx.shortest_path(G, source=skel_extrema_is[0], target=skel_extrema_is[1])
            self.midline_coords = skel_coords[c,:]

        elif n_extrema>2:
            self.good_coords = True
            # i1, i2 = self.NPCurveHandler.get_two_furthest_pts_index(skel_extrema_pts)
            # c = nx.shortest_path(G, source= skel_extrema_is[i1], target= skel_extrema_is[i2])
            # self.midline_coords = skel_coords[c,:]
            self.midline_coords = self.get_extrema_and_path_with_longest_graph_distance(n_extrema, skel_extrema_is,skel_coords, G)

        else:
            self.good_coords = False
            self.midline_coords = skel_coords

        if not self.is_midline_length_okay():
            self.good_coords = False

        return self.good_coords, self.midline_coords , n_extrema, skel_extrema_is , skel_extrema_pts
    def get_extrema_and_path_with_longest_graph_distance(self, n_extrema, skel_extrema_is,skel_coords,  G):
        all_pairs = list(itertools.combinations(list(range(n_extrema)), 2))
        max_path_length = -1
        longest_path = []
        for pair in all_pairs:

            i1, i2 = pair
            c = nx.shortest_path(G, source= skel_extrema_is[i1], target= skel_extrema_is[i2])

            coord_path = skel_coords[c, :]
            path_length = self.NPCurveHandler.get_curve_length(coord_path)
            if path_length> max_path_length:
                max_path_length = path_length
                longest_path = coord_path
        return longest_path#,max_path_length



    def init_Connectivity_graph(self,skeleton_coords):

        G = nx.Graph()

        pts = list(range((skeleton_coords.shape[0])))

        G.add_nodes_from(pts)

        centroids_tree = spatial.cKDTree(skeleton_coords)
        pairs = centroids_tree.query_pairs(r=(2**.5)+0.5)
        G.add_edges_from(pairs)

        return pts,G

    def skeletonize(self, outer_thresholded_worm):
        #outer_thresholded_worm[outer_thresholded_worm==1]=255
        #cv2.ximgproc.thinning(outer_thresholded_worm)
        outer_thresholded_worm[outer_thresholded_worm==255]=1
        skel_img = skeletonize(outer_thresholded_worm)
        #skel_img[skel_img==True] = 255
        #cv2.imwrite("thresholded.jpeg", outer_thresholded_worm)
        #cv2.imwrite("slkel.jpeg", skel_img.astype('uint8'))
        skel_pts = np.argwhere(skel_img)
        return skel_pts

    def get_length_from_midline(self):
        if self.midline_coords.size==0:
            print("warning: midline coords not yet determined")
        midline_length = self.NPCurveHandler.get_curve_length(self.midline_coords)
        return midline_length

    def sample_midline_coords_evenly(self):
        if self.midline_coords.size==0:
            print("warning: midline coords not yet determined")
        evenly_spaced_midline_pts = self.NPCurveHandler.get_even_points_along_curve(self.midline_coords, n_pts_to_sample = self.n_midline_pts)
        return evenly_spaced_midline_pts


    def is_midline_length_okay(self):
        midline_length = self.get_length_from_midline()
       # print("midlinelength", midline_length)
        if midline_length>self.midline_length_thres:
            return True
        else:
            return False