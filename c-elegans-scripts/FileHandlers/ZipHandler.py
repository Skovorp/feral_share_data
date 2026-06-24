# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 13:28:13 2022

@author: fbuck
"""
import os
import zipfile
import shutil



class ZipHandler():
    def __init__(self):
        pass

    #make zip file of folder
    def zipdir(self,path, zipfile_handle):
        # ziph is zipfile handle
        for root, dirs, files in os.walk(path):
            for file in files:
                zipfile_handle.write(os.path.join(root, file),
                           os.path.relpath(os.path.join(root, file),
                                           os.path.join(path, '..')))

    def compress_folder(self,folder_to_zip,zipped_folder_name=""):
        if zipped_folder_name == "":
            zipped_folder_name =self.get_compressed_folder_name(folder_to_zip)

        with zipfile.ZipFile(zipped_folder_name, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            #zipdir('tmp/', zipf)
            self.zipdir(folder_to_zip, zip_file)
        return zipped_folder_name

    def extract_zipfile(self,compressed_folder_name, directory_to_extract_to):

        with zipfile.ZipFile(compressed_folder_name, 'r') as zip_ref:
            zip_ref.extractall(directory_to_extract_to)

    def compress_and_delete_folder(self,folder_to_zip, zipped_folder_name=""):

        zipped_folder = self.compress_folder(folder_to_zip, zipped_folder_name = zipped_folder_name)
        shutil.rmtree(folder_to_zip)
        return zipped_folder

    def get_compressed_folder_name(self,folder_to_zip):
        return folder_to_zip[:-1]+"_compressed.zip"

#test woooooo

# folder_to_zip = os.path.join('C:\\Users\\fbuck\\Downloads\\large_plate_tracker_10252022\\histamine-2-cropped\\jpegs\\0_5')
# compressed_folder_name = "woo.zip"
# compress_folder(folder_to_zip,compressed_folder_name)

#extract_zipfile( "woo.zip",'C:\\Users\\fbuck\\Downloads\\large_plate_tracker_10252022\\' )