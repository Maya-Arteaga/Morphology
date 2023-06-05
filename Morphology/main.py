#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: juanpablomayaarteaga
"""

#Enviroment: Garza-Lab
#conda install -c conda-forge imagecodecs
#pip install opencv-python
#pip install tifffile
#pip install pandas
#pip install PySimpleGUI
#pip install matplotlib
#pip install scipy
#pip install scikit-image



#Libraries

import os
import cv2
import pandas as pd
import numpy as np
from morpho import set_path, save_tif, erase, count, count_branches, detect_and_color, gammaCorrection
import tifffile as tiff
import tkinter as tk
import PySimpleGUI as sg
from skimage import io, filters
#import matplotlib.pyplot as plt
#from tkinter import filedialog, Tk
from skimage.morphology import skeletonize





# Initial Message


layout = [[sg.Text('\n \n Hello! \n \n Previous to realize the preprocessing of your image, please store the image to be analyzed in a specific directory. \n It is recommended to store only one image per directory to avoid confunsion due to the number of images produced. \n \n Once this is done, press "Ok" and select the directory that contains your image.\n \n \n \n ', 
                    font=("Helvetica", 20, "bold"))],
          [sg.Button("Ok")]]
                       
window = sg.Window("Preprocessing to asses cell Morphology: Maya", layout)
event, values = window.read()
window.close()


#Paths 
root = tk.Tk()
root.withdraw()
i_path = tk.filedialog.askdirectory()

#i_path="/Users/juanpablomayaarteaga/Desktop/Prueba_Morfo/"

o_path= set_path(i_path+"/Output_images/")

Masks= set_path(o_path+"Masks/")

Label_path= set_path(o_path+"Label/")

Cells_path= set_path(o_path+"Cells/")

csv_doc= set_path(o_path+"Data/")

df= pd.DataFrame(columns=["Cell", "End_Points", "Junctions", "Branches", "Total_Branches_Length"])

#For loop for the files of the directory

for images in os.listdir(i_path):
    if images.endswith(".tif"):
        input_file = os.path.join(i_path, images)
        img = io.imread(input_file)
        image=img[:,:,0]
        #plt.imshow(image)
        save_tif(images, name="_original.tif", path=Masks, variable=image)
     
        
        bright_img = cv2.convertScaleAbs(image, alpha=2.5, beta=0)
        save_tif(images, name="_bright.tif", path=Masks, variable=bright_img)
        
        
        # Apply a Gaussian filter
        gaussian_array = filters.gaussian(bright_img, sigma=1.8)
        gaussian_image = (gaussian_array * 255).astype(np.uint8)
        save_tif(images, name="_gaussian.tif", path=Masks, variable=gaussian_image)
        
        
        #Gamma Correction: Non-lineal preprocessing to denoissign the image
        gammaImg = gammaCorrection(gaussian_image, 1.25)
        save_tif(images, "_gamma.tif", Masks, gammaImg)
        
        
        #Setting a threshold
        ret, thresh = cv2.threshold(gammaImg, 20, 255, cv2.THRESH_BINARY)
        save_tif(images, name="_binary.tif", path=Masks, variable=thresh)
        
        
        
        # Label objects
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=4)
        
        # Set parameters
        min_area = 500
        max_area = 1200
        num_cells_filtered = 0
        
        
        # Loop to get individual cells from threshold image
        individual_cell=0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area or area > max_area:
                labels[labels == i] = 0
           
            else:                
                bounding_box_x = stats[i, cv2.CC_STAT_LEFT]
                bounding_box_y = stats[i, cv2.CC_STAT_TOP]
                bounding_box_width = stats[i, cv2.CC_STAT_WIDTH]
                bounding_box_height = stats[i, cv2.CC_STAT_HEIGHT]
                
                #Individual cells
                object_img = thresh[bounding_box_y:bounding_box_y + bounding_box_height, bounding_box_x:bounding_box_x + bounding_box_width]
                individual_cell += 1
                #Save individual cells
                output_filename = f"{images[:-4]}_cell_{individual_cell}.tif"
                output_path = os.path.join(Cells_path, output_filename)
                tiff.imwrite(output_path, object_img)
                
                # Draw a rectangle around the cell
                cv2.rectangle(thresh, (bounding_box_x, bounding_box_y), (bounding_box_x + bounding_box_width, bounding_box_y + bounding_box_height), (255, 0, 0), 2)

        # Save the threshold image with red rectangles around objects
        save_tif(images, name="_rectangles.tif", path=Label_path, variable=thresh)
       
        
        
       
       #Loop to identify individual cells within the original image. So then, compare it.
        labeled_cell=0
        # Iterate over each labeled object
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area or area > max_area:
                labels[labels == i] = 0
                
            else:
                bounding_box_x = stats[i, cv2.CC_STAT_LEFT]
                bounding_box_y = stats[i, cv2.CC_STAT_TOP]
                bounding_box_width = stats[i, cv2.CC_STAT_WIDTH]
                bounding_box_height = stats[i, cv2.CC_STAT_HEIGHT]
                # Draw a rectangle around the object in the original image
                cv2.rectangle(img, (bounding_box_x, bounding_box_y), (bounding_box_x + bounding_box_width, bounding_box_y + bounding_box_height), (255, 255, 0), 2)
                labeled_cell += 1
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottom_left = (bounding_box_x, bounding_box_y + bounding_box_height + 20)
                font_scale = 0.5
                color = (255, 0, 0)
                thickness = 2
                cv2.putText(img, str(labeled_cell), bottom_left, font, font_scale, color, thickness)
        
        # Save the original image with red rectangles around objects
        save_tif(images, name="_original_labeled.tif", path=Label_path, variable=img)
      
        
#Skeletonize and detect branches

for cell in os.listdir(Cells_path):
    if cell.endswith(".tif"):
        input_cell = os.path.join(Cells_path, cell)
        cell_img = cv2.imread(input_cell, cv2.IMREAD_GRAYSCALE)
        scale = cell_img /255
        
        #Skeletonize
        skeleton = skeletonize(scale)
        clean_skeleton= erase(skeleton, 40)
        save_tif(cell, name="_skeletonized.tif", path=Cells_path, variable=clean_skeleton)
        
        #Detected
        M, colored_image= detect_and_color(clean_skeleton)
        save_tif(cell, name="_colored.tif", path=Cells_path, variable=colored_image)
        
       
for cell in os.listdir(Cells_path):
    if cell.endswith("_colored.tif"): 
        input_cell = os.path.join(Cells_path, cell)
        cell_img = io.imread(input_cell)
        
        End_points=cell_img[:,:,0]
        num_end_points= count(End_points)
                
        Junction_points=cell_img[:,:,1]
        num_junction_points= count(Junction_points)
        
        Length=cell_img[:,:,2]
        branches_length= count(Length)+num_end_points
                
        Branches=cell_img[:,:,2]
        num_branches= count_branches(Branches)
                
        df.loc[len(df)]=[cell, num_end_points, num_junction_points, num_branches, branches_length ]

        
df.to_csv(csv_doc+"Cell_Morphology.csv", index=False)

        
# Ending Message
layout = [[sg.Text('\n \n \n Preprocess is completed! \n \n \n Inside the selected directory, there is a folder called "Output_Images". \n \n This folder contains: \n \n A) "Masks", a folder with the preprocessed images \n \n B) "Label", a folder with the Binary image and the Original image with the selected cells for further morphological analysis \n \n C) "Cells", a folder with the individual selected cells for morphological analysis \n \n \n \n Notice that, to a better analysis, you should examine the "original_labeled" image and \n localize the correct cells for further morphological analysis \n Identify the cell number at bottom left of its rectangule. \n \n Then, look for the corrsponding cell number in the "Cells" directory and continue with the morphological analysis. \n \n \n \n   ', 
                   font=('Helvetica', 20, 'bold'))]]
window = sg.Window("Proccess done: Maya", layout)
event, values = window.read()
window.close()


        