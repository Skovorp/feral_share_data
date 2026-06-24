# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 15:30:48 2022

@author: fbuck
"""
import yaml

class YamlHandler():
    def __init__(self):
        pass

    def get_dictionary_from_yaml(self, yaml_file_name):
        stream = open(yaml_file_name,'r')
        dictionary = yaml.load(stream, yaml.FullLoader)
        return dictionary
    def generate_yaml_from_dictionary(self, dictionary, yaml_filename):
        with open(yaml_filename, 'w') as file:
            yaml.dump(dictionary, file)