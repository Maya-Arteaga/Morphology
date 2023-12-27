#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 01:01:09 2023

@author: juanpablomayaarteaga
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from morpho import set_path, save_tif
import os



i_path = "/Users/juanpablomayaarteaga/Desktop/Confocal/"
o_path = set_path(i_path + "/Output_images/")


# Open an image
images = ["1", "2", "3"]
#images = ["2"]

          
tif= ".tif"

for img in images:
    image_path = i_path + img + tif
    os.path.isfile(image_path)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


    # Check the shape of the image
    height, width, channels = image.shape
    print(f"Image shape: Height = {height}, Width = {width}, Channels = {channels}")
    
    
    # Separate channels
    r, g, b = cv2.split(image)
    
    # Display the original image and separated channels
    plt.imshow(image)
    plt.imshow(r)
    plt.imshow(g)
    
    save_tif(f"merge_{img}.tif", name=".tif", path=o_path, variable=image)
    save_tif(f"microglia_{img}.tif", name=".tif", path=o_path, variable=r)
    save_tif(f"placas_{img}.tif", name=".tif", path=o_path, variable=g)
    #plt.imshow(b)
    


