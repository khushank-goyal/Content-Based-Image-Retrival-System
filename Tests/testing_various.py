import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import skew
from scipy.spatial.distance import euclidean, cityblock
from skimage.io import imread, imshow
from skimage import feature, exposure
from sklearn import preprocessing, metrics
from PIL import Image
import sys
from xhtml2pdf import pisa
import os


# def get_X_type_dir_list(folderName = 'sample_images'):

#     directoryList = []

#     # Add all file paths under directory to list. 
#     for path in os.listdir(folderName):

#         fullPath = os.path.join(folderName, path)
#         # if (fullPath != baseImagePath):
#         #     directoryList.append(fullPath)
#         directoryList.append(fullPath)

#     return directoryList

# print(get_X_type_dir_list())


def get_X_type_dir_list(XType, folderName = 'database'):

    directoryList = []

    # Add all matching file paths under directory to list. 
    for path in os.listdir(folderName):

        fullPath = os.path.join(folderName, path)
        if (XType in fullPath):
             directoryList.append(fullPath)
        

    return directoryList

print(get_X_type_dir_list('cc'))
print(len(get_X_type_dir_list('cc')))



# listOLists = [[1,2,3], [2,3,1],[3,1,2],[1,2,3]]

# arrayStyle = np.array(listOLists)

# print(repr(arrayStyle))