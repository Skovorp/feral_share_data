import os
import sys
from pathlib import Path

class OSHandler():
    def __init__(self):
        pass
    def make_new_folder(self, parent_dir, new_folder_name):
        path = os.path.join(parent_dir, new_folder_name)
        try:
            os.makedirs(path, exist_ok = True)
            return path
        except:
            return path

    def get_immediate_subdirectory_names_and_paths(self, parent_dir):
        child_folder_name_and_paths = [(f.name, os.path.join(f.path,"")) for f in os.scandir(parent_dir) if f.is_dir()]

        return child_folder_name_and_paths

    def get_parent_dir(self,file_or_folder):
        path = Path(file_or_folder)
        return path.parent.absolute()
import os, sys

class HiddenPrints:
    #from https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout