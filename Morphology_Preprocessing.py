#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: juanpablomayaarteaga
"""

#Libraries

from skimage import io, filters
import os
import cv2
import matplotlib.pyplot as plt
import tifffile as tiff
import numpy as np
import tkinter as tk
from tkinter import filedialog
import PySimpleGUI as sg







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
i_path = filedialog.askdirectory()

#i_path="/Users/juanpablomayaarteaga/Desktop/Output_Images/Pruebas/Iba/"


o_path=i_path+"/Output_images/"
if not os.path.isdir(o_path):
    os.mkdir(o_path)


Masks=o_path+"Masks/"
if not os.path.isdir(Masks):
    os.mkdir(Masks)
    
    
Label_path=o_path+"Label/"
if not os.path.isdir(Label_path):
    os.mkdir(Label_path)





#For loop for the files of the directory

for filename in os.listdir(i_path):
    if filename.endswith(".tif"):
        input_file = os.path.join(i_path, filename)
        img = io.imread(input_file)
        image=img[:,:,0]
        #plt.imshow(image)
        
        # Save the Normal image
        output_filename = filename[:-4] + "_normal.tif"
        output_path = os.path.join(Masks, output_filename)
        tiff.imwrite(output_path, image)
     
        
        bright_img = cv2.convertScaleAbs(image, alpha=2.5, beta=0)
        # Save the Brigther image
        output_filename = filename[:-4] + "_bright12.tif"
        output_path = os.path.join(Masks, output_filename)
        tiff.imwrite(output_path, bright_img)
        
        
        
        # Apply a Gaussian filter
        smooth_radius = filters.gaussian(bright_img, sigma=1.8)
        Gaussian3 = (smooth_radius * 255).astype(np.uint8)
        # Save the Gaussian image
        output_filename = filename[:-4] + "_Gaussian10.tif"
        output_path = os.path.join(Masks, output_filename)
        tiff.imwrite(output_path, Gaussian3)
        
        
        
        #Gamma Correction: Non-lineal preprocessing to denoissign the image
        
        def gammaCorrection(src, gamma):
            invGamma = 1 / gamma
        
            table = [((i / 255) ** invGamma) * 255 for i in range(256)]
            table = np.array(table, np.uint8)
        
            return cv2.LUT(src, table)
        
        gammaImg = gammaCorrection(Gaussian3, 0.8)
        ######SAVE Gamma
        output_filename = filename[:-4] + "_gamma3.tif"
        output_path = os.path.join(Masks, output_filename)
        tiff.imwrite(output_path, gammaImg)
        
        
        
        
        #Setting a threshold
        ret, thresh = cv2.threshold(gammaImg, 20, 255, cv2.THRESH_BINARY)
        ######SAVE Thresh
        output_filename = filename[:-4] + "_threshold.tif"
        output_path = os.path.join(Masks, output_filename)
        tiff.imwrite(output_path, thresh)
        
        
        
        
        
        # Label objects
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=4)
        
        # Set threshold for object area
        min_area = 500
        max_area = 1200
        num_cells_filtered = 0
        
        # Save Cropped cells into a directory
        output_dir = o_path + "Cells"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        
        # Getting cells
        cropped_cell=0
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
                # Save the cropped image with filename as "filename_object_valid_object"
                cropped_cell += 1
                output_filename = f"{filename[:-4]}_cell_{cropped_cell}.tif"
                output_path = os.path.join(output_dir, output_filename)
                tiff.imwrite(output_path, object_img)
                
                
                
                
                # Draw a rectangle around the cell
                cv2.rectangle(thresh, (bounding_box_x, bounding_box_y), (bounding_box_x + bounding_box_width, bounding_box_y + bounding_box_height), (255, 0, 0), 2)

        
        # Save the threshold image with red rectangles around objects
        output_filename = filename[:-4] + "_rectangles.tif"
        output_path = os.path.join(Label_path, output_filename)
        tiff.imwrite(output_path, thresh)



       
       
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
        output_filename = filename[:-4] + "_original_labeled.tif"
        output_path = os.path.join(Label_path, output_filename)
        tiff.imwrite(output_path, img)
     
     
        
# Ending Message
layout = [[sg.Text('\n \n \n Preprocess is completed! \n \n \n Inside the selected directory, there is a folder called "Output_Images". \n \n This folder contains: \n \n A) "Masks", a folder with the preprocessed images \n \n B) "Label", a folder with the Binary image and the Original image with the selected cells for further morphological analysis \n \n C) "Cells", a folder with the individual selected cells for morphological analysis \n \n \n \n Notice that, to a better analysis, you should examine the "original_labeled" image and \n localize the correct cells for further morphological analysis \n Identify the cell number at bottom left of its rectangule. \n \n Then, look for the corrsponding cell number in the "Cells" directory and continue with the morphological analysis. \n \n \n \n   ', 
                   font=('Helvetica', 20, 'bold'))]]
window = sg.Window("Proccess done: Maya", layout)
event, values = window.read()
window.close()
        
  
        
