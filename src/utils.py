# general helper functions

import os

def makedirs(path_list):
    for path in path_list:
        if not os.path.isdir(path):
            os.makedirs(path)