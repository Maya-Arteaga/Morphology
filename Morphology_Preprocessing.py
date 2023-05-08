#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  7 21:25:35 2023

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







#0) Initial Message




layout = [[sg.Text('\n \n   ¡Hola! \n \n Para realizar el preprocesamiento del análisis de morfología, sigue los siguientes pasos:  \n \n 1) Previo a correr el programa, coloca en una carpeta la imágen que quieres preprocesar \n \n 2) Selecciona la carpeta que contiene la imágen.\n \n \n \n ', 
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

        # Save the Otsu image
        output_filename = filename[:-4] + "_bright12.tif"
        output_path = os.path.join(Masks, output_filename)
        tiff.imwrite(output_path, bright_img)
        
        
        
        #3) Apply a Gaussian filter
        smooth_radius = filters.gaussian(bright_img, sigma=1.8)
        Gaussian3 = (smooth_radius * 255).astype(np.uint8)

        # Save the filtered image
        output_filename = filename[:-4] + "_Gaussian10.tif"
        output_path = os.path.join(Masks, output_filename)
        tiff.imwrite(output_path, Gaussian3)
        
        #B)Gamma Correction: Non-lineal preprocessing to denoissign the image
        
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
        
        #D)Setting a threshold
        ret, thresh = cv2.threshold(gammaImg, 20, 255, cv2.THRESH_BINARY)
        ######SAVE Thresh
        output_filename = filename[:-4] + "_threshold.tif"
        output_path = os.path.join(Masks, output_filename)
        tiff.imwrite(output_path, thresh)
        
        
        
        
        
        # Perform connected component analysis and label objects
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=4)
        
        # Set threshold for object area
        min_area = 500
        max_area = 1200
        
        # Create directory to save cropped images
        output_dir = o_path+"Cropped_objects"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Iterate over each labeled object
        for i in range(1, num_labels):
        
            # Get statistics for object i
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                labels[labels == i] = 0
            elif area > miax_area:
                labels[labels == i] = 0
            else:
                bounding_box_x = stats[i, cv2.CC_STAT_LEFT]
                bounding_box_y = stats[i, cv2.CC_STAT_TOP]
                bounding_box_width = stats[i, cv2.CC_STAT_WIDTH]
                bounding_box_height = stats[i, cv2.CC_STAT_HEIGHT]
        
                # Crop the rectangular region corresponding to the object
                object_img = thresh[bounding_box_y:bounding_box_y+bounding_box_height, bounding_box_x:bounding_box_x+bounding_box_width]
        
                # Save the cropped image
                output_filename = f"object_{i}.tif"
                output_path = os.path.join(output_dir, output_filename)
                tiff.imwrite(output_path, object_img)
        
                # Draw a red rectangle around the object in the original image
                cv2.rectangle(thresh, (bounding_box_x, bounding_box_y), (bounding_box_x + bounding_box_width, bounding_box_y + bounding_box_height), (255, 0, 0), 2)
        
        # Save the threshold image with red rectangles around objects
        output_filename = filename[:-4] + "_rectangles.tif"
        output_path = os.path.join(Label_path, output_filename)
        tiff.imwrite(output_path, thresh)
                
        
        # Iterate over each labeled object
        for i in range(1, num_labels):
        
            # Get statistics for object i
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                labels[labels == i] = 0
            elif area > miax_area:
                labels[labels == i] = 0
            else:
                bounding_box_x = stats[i, cv2.CC_STAT_LEFT]
                bounding_box_y = stats[i, cv2.CC_STAT_TOP]
                bounding_box_width = stats[i, cv2.CC_STAT_WIDTH]
                bounding_box_height = stats[i, cv2.CC_STAT_HEIGHT]
        
                # Crop the rectangular region corresponding to the object
                #object_img = img[bounding_box_y:bounding_box_y+bounding_box_height, bounding_box_x:bounding_box_x+bounding_box_width]
        
                # Save the cropped image
               # output_filename = f"object_{i}.tif"
                #output_path = os.path.join(output_dir, output_filename)
                #tiff.imwrite(output_path, object_img)
        
                # Draw a red rectangle around the object in the original image
                cv2.rectangle(img, (bounding_box_x, bounding_box_y), (bounding_box_x + bounding_box_width, bounding_box_y + bounding_box_height), (255, 255, 0), 2)
                
                
                # Add a label with the object number
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottom_left = (bounding_box_x, bounding_box_y + bounding_box_height + 20)
                font_scale = 0.5
                color = (255, 0, 0)
                thickness = 2
                cv2.putText(img, str(i), bottom_left, font, font_scale, color, thickness)
        
        
        # Save the original image with red rectangles around objects
        output_filename = filename[:-4] + "_original_labeled.tif"
        output_path = os.path.join(Label_path, output_filename)
        tiff.imwrite(output_path, img)
        
        
# Message
layout = [[sg.Text('\n \n \n ¡Listo! \n \n El preprocesamiento ha terminado. \n \n Dentro de la carpeta que seleccionaron en el paso (2) se encontrará la carpeta "Output_Images". \n Esa carpeta contendrá: \n \n A) "Masks", una carpeta con el procesamiento que se hizo a las imágenes \n \n B) "Label", una carpeta que contiene las células identificadas \n \n C) "Cropped_Obects", una carpeta que contiene las celulas recortadas \n \n \n \n', 
                   font=('Helvetica', 20, 'bold'))]]
window = sg.Window("Proccess done: Maya", layout)
event, values = window.read()
window.close()
        
  
        
