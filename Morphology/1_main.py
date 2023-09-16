#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 13:12:20 2023

@author: juanpablomayaarteaga

En este codigo se intenta realizar un analisis de morfologia mejorado de la version de Zhan usada en imageJ

"""


#Enviroment: Garza-Lab
#conda activate Garza-Lab
#conda install -c conda-forge imagecodecs
#pip install opencv-python
#pip install tifffile
#pip install pandas
#pip install PySimpleGUI
#pip install matplotlib
#pip install scipy
#pip install scikit-image
#pip install plotly



#Libraries

import os
import cv2
import pandas as pd
import numpy as np
from morpho import set_path, save_tif, erase, count, count_branches, detect_and_color, gammaCorrection, calculate_area, find_contours, polygon, detect_features
import tifffile as tiff
import tkinter as tk
import PySimpleGUI as sg
from skimage import io, filters
#import matplotlib.pyplot as plt
#from tkinter import filedialog, Tk
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from skimage import restoration
import re
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
from scipy.spatial import distance
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Initial Message

"""
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
"""


"""
ESTABLECER LOS PATHS QUE SE UTILIZARÁN Y CREARÁN

"""

i_path="/Users/juanpablomayaarteaga/Desktop/Prueba_Morfo/"


o_path= set_path(i_path+"/Output_images/")

Preprocess_path= set_path(o_path+"Preprocess/")

ID_path= set_path(o_path+"ID/")

Cells_path= set_path(o_path+"Cells/")

Cells_original_path= set_path(Cells_path+"Cells_original/")

Therhold_path= set_path(Cells_path+"Thershold/")

Skeleton_path= set_path(Cells_path+"Skeleton/")

Soma_path= set_path(Cells_path+"Soma/")

Branches_path= set_path(Cells_path+"Branches/")

Skeleton2_path= set_path(Cells_path+"Skeleton2/")

Branches2_path= set_path(Cells_path+"Branches2/")

Branches3_path= set_path(Cells_path+"Branches3/")

Skeleton_Soma_path= set_path(Cells_path+"Skeleton_Soma/")

Polygon_path= set_path(Cells_path+"Polygon/")

Polygon_Centroid_path= set_path(Cells_path+"Polygon_Centroid/")

Contour_path = set_path(Cells_path+"Contour/")

Soma_centroid_path= set_path(Cells_path+"Soma_Centroids/")

csv_path= set_path(o_path+"Data/")


Plot_path= set_path(o_path+"Plots/")



#SE ESTABLECE QUE COLUMNAS TENDRA NUESTRO DATAFRAME 
df= pd.DataFrame(columns=["Cell", "End_Points", "Junctions", "Branches", "Total_Branches_Length"])



#For loop for the files of the directory
#AQUI SE HACE UN LOOP PARA REALIZARLE A TODAS LAS IMAGENES DE UN DIRECTORIO, LA MISMA SECUENCIA DE PASOS
#AQUI EMPIEZA EL PREPROCESAMIENTO PARA AFINAR LOS DETALLES DE CADA CELULAR Y PREPARARLAS PARA SU ANALISIS

for images in os.listdir(i_path):
    if images.endswith(".tif"):
        
        #LEER LA IMAGEN
        input_file = os.path.join(i_path, images)
        img = io.imread(input_file)
        
        print ("...")
        print ("...")
        print ("Reading image...")
        print ("...")
        print ("...")
        
        #SEPARAR CANALES. NOS QUEDAMOS SOLO CON LA MICROGLIA
        image=img[:,:,0]
        #plt.imshow(image)
        save_tif(images, name="_original.tif", path=Preprocess_path, variable=image)
     
        print ("...")
        print ("...")
        print ("Getting Microglia...")
        print ("...")
        print ("...")
        
        bright_img = cv2.convertScaleAbs(image, alpha=3, beta=0)
        #plt.imshow(bright_img)
        save_tif(images, name="_bright.tif", path=Preprocess_path, variable=bright_img)
        
        print ("...")
        print ("...")
        print ("Enhencing image brigthness...")
        print ("...")
        print ("...")
        
        # Apply a Gaussian filter
        gaussian_array = filters.gaussian(bright_img, sigma=2.2)
        gaussian_image = (gaussian_array * 255).astype(np.uint8)
        #plt.imshow(gaussian_image)
        save_tif(images, name="_gaussian.tif", path=Preprocess_path, variable=gaussian_image)
        
        print ("...")
        print ("...")
        print ("Enhencing image smothness...")
        print ("...")
        print ("...")
        
        
        #Gamma Correction: Non-lineal preprocessing to denoissign the image
        gammaImg = gammaCorrection(gaussian_image, 1.25)
        #plt.imshow(gammaImg)
        save_tif(images, "_gamma.tif", Preprocess_path, gammaImg)
        
        
        print ("...")
        print ("...")
        print ("Enhencing image illumination...")
        print ("...")
        print ("...")
        
        #Setting a threshold. "RET"  GUARDA EL VALOR DEL THRESHOLDING
        ret, thresh = cv2.threshold(gammaImg, 20, 255, cv2.THRESH_BINARY)
        #plt.imshow(thresh)
        save_tif(images, name="_binary.tif", path=Preprocess_path, variable=thresh)
        
        print ("...")
        print ("...")
        print("Preprocessing is completed!")
        print ("...")
        print ("...")

#AQUI SE TERMINAR EL PREPROCESAMIENTO,     
        
#CRITERIOS PARA DETECTAR LOS OBJETOS. SE USAN PARA ELIMINAR PARTICULAS QUE NO CORRESPONDAN A CELULAS        
        # Label objects
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=4)
        
        # Set parameters
        min_area = 500
        max_area = 2400
        #max_area=6000
        num_cells_filtered = 0
        
        

        
  
#### De aqui se sustraen las células individuales de la imagen original para observarlas
#Conservan el nombre de la imagen original y su numero de identificación
        print ("...")
        print ("...")
        print("Identifying individual cells...")
        print ("...")
        print ("...")
        # Loop to get individual cells from threshold image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert image to BGR color format
        #plt.imshow(image)

        individual_cell=0
        # Initialize the list to store cell positions
        cell_positions = []
        cell_num = []
        
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area or area > max_area:
                labels[labels == i] = 0
           
            else:                
                bounding_box_x = stats[i, cv2.CC_STAT_LEFT]
                bounding_box_y = stats[i, cv2.CC_STAT_TOP]
                bounding_box_width = stats[i, cv2.CC_STAT_WIDTH]
                bounding_box_height = stats[i, cv2.CC_STAT_HEIGHT]
                cell_positions.append((bounding_box_x, bounding_box_y, bounding_box_x + bounding_box_width, bounding_box_y + bounding_box_height))
                
                #Individual cells
                object_img = image[bounding_box_y:bounding_box_y + bounding_box_height, bounding_box_x:bounding_box_x + bounding_box_width]
                individual_cell += 1
                cell_num.append(individual_cell)
                #Save individual cells   
                output_filename = f"{images[:-4]}_cell_{individual_cell}.tif"
                output_path = os.path.join(Cells_original_path, output_filename)
                tiff.imwrite(output_path, object_img)
                
                
                
                # Draw a rectangle around the cell
                #cv2.rectangle(image, (bounding_box_x, bounding_box_y), (bounding_box_x + bounding_box_width, bounding_box_y + bounding_box_height), (255, 0, 0), 2)

        # Save the threshold image with red rectangles around objects
        #save_tif(images, name="_rectangles.tif", path=Label_path, variable=image)
        print ("...")
        print ("...")
        print("Individual cells identified")
        print ("...")
        print ("...")
              
        



        print ("...")
        print ("...")
        print("Thersholding individual cells...")
        print ("...")
        print ("...")
        ###### Thershold of the cells
        
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
                output_path = os.path.join(Therhold_path, output_filename)
                tiff.imwrite(output_path, object_img)
                
                # Draw a rectangle around the cell
                #cv2.rectangle(thresh, (bounding_box_x, bounding_box_y), (bounding_box_x + bounding_box_width, bounding_box_y + bounding_box_height), (255, 0, 0), 2)

        # Save the threshold image with red rectangles around objects
        #save_tif(images, name="_rectangles.tif", path=Label_path, variable=thresh)
       
        print ("...")
        print ("...")
        print("Binary Mask of individual cells ready!")
        print ("...")
        print ("...")
        
        
        print ("...")
        print ("...")
        print("Generating reference image of cell identified...")
        print ("...")
        print ("...")
       

#Loop to identify individual cells within the original image. So then, compare it.
       
        color_image = image
        labeled_cell2 = 0
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
                cv2.rectangle(color_image, (bounding_box_x, bounding_box_y), (bounding_box_x + bounding_box_width, bounding_box_y + bounding_box_height), (255, 255, 0), 2)
                labeled_cell2 += 1
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottom_left = (bounding_box_x, bounding_box_y + bounding_box_height + 20)
                font_scale = 0.5
                color = (255, 0, 0)
                thickness = 2
                cv2.putText(color_image, str(labeled_cell2), bottom_left, font, font_scale, color, thickness)
        
        # Save the original image with red rectangles around objects
        save_tif(images, name="_microglia_identified.tif", path=ID_path, variable=color_image)


        print ("...")
        print ("...")
        print("Reference image ready!")
        print ("...")
        print ("...")

        print ("...")
        print ("...")
        print("Generating reference image of cells preprocessed...")
        print ("...")
        print ("...")

       #Loop to identify individual cells within the original image. So then, compare it.
        color_gammaimage = cv2.cvtColor(gammaImg, cv2.COLOR_GRAY2BGR)
        labeled_cell2 = 0
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
                cv2.rectangle(color_gammaimage, (bounding_box_x, bounding_box_y), (bounding_box_x + bounding_box_width, bounding_box_y + bounding_box_height), (255, 255, 0), 2)
                labeled_cell2 += 1
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottom_left = (bounding_box_x, bounding_box_y + bounding_box_height + 20)
                font_scale = 0.5
                color = (255, 0, 0)
                thickness = 2
                cv2.putText(color_gammaimage, str(labeled_cell2), bottom_left, font, font_scale, color, thickness)
        
        # Save the original image with red rectangles around objects
        save_tif(images, name="_gamma_microglia_identified.tif", path=Preprocess_path, variable=color_gammaimage)
        plt.imshow(image)

        print ("...")
        print ("...")
        print("Preproccesed image with identified cells ready!")
        print ("...")
        print ("...")












###### DataFrame Cell num and cell position

df_positions = pd.DataFrame({'Cell': cell_num, 'cell_positions': cell_positions})

# Convert the "Cell" column to string type
df_positions['Cell'] = df_positions['Cell'].astype(str)









### Sustrcción del SOMA Celular

print ("...")
print ("...")
print("Substracting soma of the cells...")
print ("...")
print ("...")

for images in os.listdir(Cells_original_path):
    if images.endswith(".tif"):
        input_file = os.path.join(Cells_original_path, images)
        img = io.imread(input_file)
        img=img[:,:,0]
        plt.imshow(img)
     
        bright_img = cv2.convertScaleAbs(img, alpha=0.23, beta=0)
        
        # Non-local Means Denoising
        #denoised_image = restoration.denoise_nl_means(bright_img, patch_size=5, patch_distance=7, h=0.1)
        #denoised_image = (denoised_image * 255).astype(np.uint8) 
        #plt.imshow(denoised_image)
        
        
        #C)Opening preprocessing to dealing with the branches
        kernel = np.ones((3,3),np.uint8)
        #opening = cv2.morphologyEx(denoised_image,cv2.MORPH_OPEN,kernel, iterations = 1)
        
        #dilated = cv2.dilate(opening, kernel, iterations=1)
        
        # Apply a Gaussian filter
        gaussian_array = filters.gaussian(bright_img, sigma=2.2)
        gaussian_image = (gaussian_array * 255).astype(np.uint8)
        
        dilated = cv2.dilate(gaussian_image, kernel, iterations=1)

        
        
        #Setting a threshold
        ret, thresh = cv2.threshold(dilated, 20, 255, cv2.THRESH_BINARY)
        save_tif(images, name="_soma_.tif", path=Soma_path, variable=thresh)
        
        

print ("...")
print ("...")
print("Soma substracted!")
print ("...")
print ("...")


print ("...")
print ("...")
print("Substracting Soma features...")
print ("...")
print ("...")


#GENERANDO EL DATAFRAME DEL SOMA CELULAR 


df_area = pd.DataFrame(columns=["Cell", "Area_soma", "Perimeter_soma", "Circularity_soma", "soma_compactness", "soma_orientation", "soma_feret_diameter", "soma_eccentricity", "soma_aspect_ratio"])  # Initialize DataFrame with column names

# Cell Area, Perimeter, and Circularity
for soma in os.listdir(Soma_path):
    if soma.endswith(".tif"):
        input_cell = os.path.join(Soma_path, soma)
        area = calculate_area(input_cell)
        
        if area != 0:
            contours = find_contours(input_cell)  
            
            soma_contour = max(contours, key=cv2.contourArea)
            
            # Calculate the eccentricity
            if len(soma_contour) >= 5:
                _, (major_axis, minor_axis), _ = cv2.fitEllipse(soma_contour)
                soma_eccentricity = major_axis / minor_axis
            
            
            # Calculate the bounding box of the contour
            x, y, width, height = cv2.boundingRect(soma_contour)
            
            # Calculate the aspect ratio
            soma_aspect_ratio = width / float(height)
            
            
            # Calculate the Feret diameter (maximum caliper)
            min_rect = cv2.minAreaRect(soma_contour)
            soma_feret_diameter = max(min_rect[1])  # Maximum of width and height
            
            # Calculate the orientation (angle of the Feret diameter)
            soma_orientation = min_rect[2]  # Angle in degrees
            
            # Calculate the perimeter
            perimeter = 0
            for contour in contours:
                perimeter += cv2.arcLength(contour, closed=True) 
            
            # Calculate circularity and compactness only if the perimeter is non-zero
            if perimeter != 0:
                circularity = 4 * np.pi * (area / (perimeter ** 2))
                soma_compactness = area / perimeter
                
            
        else:
            area = 0
            soma_feret_diameter = 0
            soma_orientation = 0
            perimeter = 0
            circularity = 0 
            soma_compactness = 0
            soma_eccentricity = 0
            soma_aspect_ratio = 0
            
        
        
        df_area.loc[len(df_area)] = [soma, area, perimeter, circularity, soma_compactness, soma_orientation, soma_feret_diameter, soma_eccentricity, soma_aspect_ratio]

# Extract the numeric part from the "Cell" column and convert it to integers
df_area['Cell'] = df_area['Cell'].str.extract(r'(\d+)').astype(int)

# Sort the DataFrame by the "Cell" column in ascending order
df_area = df_area.sort_values(by='Cell')


df_area['Cell'] = df_area['Cell'].astype(str)


df_area.to_csv(csv_path + "Soma.csv", index=False)













print ("...")
print ("...")
print("Soma features substracted!")
print ("...")
print ("...")


#########################################################


print ("...")
print ("...")
print("Substracting Cell Ramifications features...")
print ("...")
print ("...")



# ANALISIS TRADICIONAL DE ZHAN DE SKELETONIZE QUE SE USA EN IMAGEJ PARA MORFOLOGIA
       
#Skeletonize and detect branches

for cell in os.listdir(Therhold_path):
    if cell.endswith(".tif"):
        input_cell = os.path.join(Therhold_path, cell)
        cell_img = cv2.imread(input_cell, cv2.IMREAD_GRAYSCALE)
        scale = cell_img /255
        
        #Skeletonize
        skeleton = skeletonize(scale)
        clean_skeleton= erase(skeleton, 40)
        save_tif(cell, name="_skeleton_.tif", path=Skeleton_path, variable=clean_skeleton)
        
        #Detecteding Branches features
        M, colored_image= detect_and_color(clean_skeleton)
        save_tif(cell, name="_traditional_.tif", path=Branches_path, variable=colored_image)
        
       
for cell in os.listdir(Branches_path):
    if cell.endswith(".tif"): 
        input_cell = os.path.join(Branches_path, cell)
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
        




df['Cell'] = df['Cell'].str.extract(r'(\d+)')        
df.to_csv(csv_path+"Cell_Traditional_Morphology.csv", index=False)


print ("...")
print ("...")
print("Cell Ramifications features substracted!")
print ("...")
print ("...")


print ("...")
print ("...")
print("Renaming images to match")
print ("...")
print ("...")




# Renaming images to match each other (to substrate de nucleus)

for filename in os.listdir(Skeleton_path):
    if filename.endswith(".tif"):
        original_path = os.path.join(Skeleton_path, filename)
        new_filename = re.findall(r"\d+", filename)[0] + ".tif"  # Extract the number and create the new filename
        new_path = os.path.join(Skeleton_path, new_filename)
        os.rename(original_path, new_path)  # Rename the file

for filename in os.listdir(Soma_path):
    if filename.endswith(".tif"):
        original_path = os.path.join(Soma_path, filename)
        new_filename = re.findall(r"\d+", filename)[0] + ".tif"  # Extract the number and create the new filename
        new_path = os.path.join(Soma_path, new_filename)
        os.rename(original_path, new_path)  # Rename the file
        
        
for filename in os.listdir(Branches_path):
    if filename.endswith(".tif"):
        original_path = os.path.join(Branches_path, filename)
        new_filename = re.findall(r"\d+", filename)[0] + ".tif"  # Extract the number and create the new filename
        new_path = os.path.join(Branches_path, new_filename)
        os.rename(original_path, new_path)  # Rename the file        

for filename in os.listdir(Polygon_path):
    if filename.endswith(".tif"):
        original_path = os.path.join(Polygon_path, filename)
        new_filename = re.findall(r"\d+", filename)[0] + ".tif"  # Extract the number and create the new filename
        new_path = os.path.join(Polygon_path, new_filename)
        os.rename(original_path, new_path)  # Rename the file
        
         
for filename in os.listdir(Polygon_Centroid_path):
    if filename.endswith(".tif"):
        original_path = os.path.join(Polygon_Centroid_path, filename)
        new_filename = re.findall(r"\d+", filename)[0] + ".tif"  # Extract the number and create the new filename
        new_path = os.path.join(Polygon_Centroid_path, new_filename)
        os.rename(original_path, new_path)  # Rename the file

for filename in os.listdir(Cells_original_path):
    if filename.endswith(".tif"):
        original_path = os.path.join(Cells_original_path, filename)
        new_filename = re.findall(r"\d+", filename)[0] + ".tif"  # Extract the number and create the new filename
        new_path = os.path.join(Cells_original_path, new_filename)
        os.rename(original_path, new_path)  # Rename the file


print ("...")
print ("...")
print("Images renamed")
print ("...")
print ("...")




# GEETTING THE SKELETON WITHOUT THE SOMA

for image in os.listdir(Skeleton_path):
    if image.endswith(".tif"):
        input_skeleton = os.path.join(Skeleton_path, image)
        input_soma = os.path.join(Soma_path, image)

        skeleton_img = io.imread(input_skeleton, as_gray=True)  # Ensure grayscale
        soma_img = io.imread(input_soma, as_gray=True)  # Ensure grayscale

        # Check if the images have the same dimensions
        if skeleton_img.shape == soma_img.shape:
            subtracted_image = cv2.subtract(skeleton_img, soma_img)
            save_tif(image, "_branch_.tif", Skeleton2_path, subtracted_image)

            # Detecting branches features
            M, colored_image = detect_and_color(subtracted_image)
            save_tif(image, name="_new_.tif", path=Branches2_path, variable=colored_image)

            added_image = cv2.add(skeleton_img, soma_img)
            save_tif(image, "_cell_.tif", Skeleton_Soma_path, added_image)

 

           
            
for filename in os.listdir(Skeleton_Soma_path):
    if filename.endswith(".tif"):
        original_path = os.path.join(Skeleton_Soma_path, filename)
        new_filename = re.findall(r"\d+", filename)[0] + ".tif"  # Extract the number and create the new filename
        new_path = os.path.join(Skeleton_Soma_path, new_filename)
        os.rename(original_path, new_path)  # Rename the file   
        
        

# CONTOUR of the cell: celula con nucleo pero hueco 


for image in os.listdir(Skeleton_Soma_path):
    if image.endswith(".tif"):
        input_file = os.path.join(Skeleton_Soma_path, image)
        img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)

        # Find the contours in the binary image
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a blank image with the same shape as the binary image
        polygon_cell = np.zeros_like(img, dtype=np.uint8)

        # Draw the contours on the blank image
        cv2.drawContours(polygon_cell, contours, -1, (255), thickness=1)  # Draw contours with a thickness of 1

        # Save the resulting polygon image with the same filename
        save_tif(image, "_contour_.tif", Contour_path, polygon_cell)






#·································GEOMETRY······················

print ("...")
print ("...")
print("Getting Cell centroids...")
print ("...")
print ("...")


centroids_cell = []

for images in os.listdir(Skeleton_Soma_path):
    if images.endswith(".tif"):
        input_file = os.path.join(Skeleton_Soma_path, images)
        img = io.imread(input_file)
        
        # Find contours in the binary image
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            # Take the first (largest) contour
            largest_contour = max(contours, key=cv2.contourArea)
        
            # Calculate the moments of the contour
            M = cv2.moments(largest_contour)
        
            if M['m00'] != 0:
                # Calculate the centroid coordinates
                centroid_x = int(M['m10'] / M['m00'])
                centroid_y = int(M['m01'] / M['m00'])
                centroid = (centroid_x, centroid_y)
                centroids_cell.append((centroid_x, centroid_y))
                
        
                print(f"Centroid coordinates: {centroid_x}, {centroid_y}")
            else:
                print("Object has no area (m00=0)")
        else:
            print("No contours found in the image")
            
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Optionally, you can draw the centroid on the image for visualization
        centroid_img= cv2.circle(img_color, centroid, 1, (255, 0, 0), -1)  # Draw a red circle at the centroid
        save_tif(images, name="_centroids.tif", path=Soma_centroid_path, variable=centroid_img)
        #plt.imshow(centroid_img)


print ("...")
print ("...")
print("Cell centroids added")
print ("...")
print ("...")




print ("...")
print ("...")
print("Generating POLYGONS...")
print ("...")
print ("...")

        
        

#POLYGON from skeleton: puede tener sesgos porque solo se basa en el skeletonize

polygon_areas = []
polygon_perimeters = []
polygon_compactness = []

for image in os.listdir(Skeleton_path):
    if image.endswith(".tif"):
        input_file = os.path.join(Skeleton_path, image)
        img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)

        # Create a polygon covering the entire cell including ramifications
        polygon_image = polygon(img)
        
        # Calculate the area using countNonZero
        area = cv2.countNonZero(polygon_image)
        polygon_areas.append(area)
        
        # Calculate the perimeter using the findContours method
        contours, _ = cv2.findContours(polygon_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = cv2.arcLength(contours[0], True)
        polygon_perimeters.append(perimeter)
        
        # Calculate the compactness
        compactness = area / perimeter
        polygon_compactness.append(compactness)

        # Save the resulting polygon image with the same filename
        save_tif(image, "_polygon_.tif", Polygon_path, polygon_image)
        



print ("...")
print ("...")
print("POLYGONS extracted")
print ("...")
print ("...")

print ("...")
print ("...")
print("Getting Polygon centroids...")
print ("...")
print ("...")


centroids_polygon = []
eccentricities = []
feret_diameters = []
orientations = []

for image in os.listdir(Polygon_path):
    if image.endswith(".tif"): 
        input_polygon = os.path.join(Polygon_path, image)
        polygon_img = io.imread(input_polygon)       

        M = cv2.moments(polygon_img)
        centroid_x = int(M['m10'] / M['m00'])
        centroid_y = int(M['m01'] / M['m00'])
        centroid = (centroid_x, centroid_y)
        
        centroids_polygon.append((centroid))


        # Find the contours of the polygon
        contours, _ = cv2.findContours(polygon_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygon_contour = max(contours, key=cv2.contourArea)
        
        # Calculate the eccentricity
        _, (major_axis, minor_axis), _ = cv2.fitEllipse(polygon_contour)
        eccentricity = major_axis / minor_axis
        eccentricities.append(eccentricity)
        
        # Calculate the Feret diameter (maximum caliper)
        min_rect = cv2.minAreaRect(polygon_contour)
        feret_diameter = max(min_rect[1])  # Maximum of width and height
        feret_diameters.append(feret_diameter)
        
        # Calculate the orientation (angle of the Feret diameter)
        orientation = min_rect[2]  # Angle in degrees
        orientations.append(orientation)

        # Approximate the polygon to get its vertices
        epsilon = 0.02 * cv2.arcLength(polygon_contour, True)
        vertices = cv2.approxPolyDP(polygon_contour, epsilon, True)

        # Draw the centroid (red) and vertices (green) on the image
        polygon_cetroid_img = cv2.cvtColor(polygon_img, cv2.COLOR_GRAY2BGR)
        cv2.circle(polygon_cetroid_img, centroid, 3, (0, 0, 255), -1)  # Red circle for centroid
        for vertex in vertices:
            x, y = vertex[0]
            cv2.circle(polygon_cetroid_img, (x, y), 2, (0, 255, 0), -1)  # Green circle for vertices

        # Save the image with visualizations
        save_tif(image, ".tif", Polygon_Centroid_path, polygon_cetroid_img)
        
        

print ("...")
print ("...")
print("Polygon centroids added")
print ("...")
print ("...")




print ("...")
print ("...")
print("Generating INITIAL POINTS...")
print ("...")
print ("...")

#Detectando  INITIAL POINTS

for image in os.listdir(Skeleton_Soma_path):
    if image.endswith(".tif"):
        input_skeleton = os.path.join(Skeleton_Soma_path, image)
        input_soma = os.path.join(Soma_path, image)

        skeleton_img = io.imread(input_skeleton, as_gray=True)
        soma_img = io.imread(input_soma, as_gray=True)

        # Check if the images have the same dimensions
        if skeleton_img.shape == soma_img.shape:
            subtracted_image = cv2.subtract(skeleton_img, soma_img)
            #save_tif(image, "_branch_.tif", Branches3_path, subtracted_image)

            #Detecting branches features
            M, colored_image3 = detect_features(subtracted_image, soma_img)
            save_tif(image, name="_new_.tif", path=Branches3_path, variable=colored_image3)
        


df_branches = pd.DataFrame(columns=["Cell", "End_Points", "Junctions", "Branches", "Initial_Points", "Total_Branches_Length"])  # Initialize DataFrame with column names


# CONTANDO LAS CARACTERISTICAS DE LAS RAMIFICACIONES PARA EL DATAFRAME



print ("...")
print ("...")
print("Analyzing the BRANCHES features...")
print ("...")
print ("...")


for cell in os.listdir(Branches3_path):
    if cell.endswith(".tif"): 
        input_cell = os.path.join(Branches3_path, cell)
        cell_img = io.imread(input_cell)
        #plt.imshow(cell_img)
                
        End_points = cell_img[:, :, 0] == 255
        num_end_points = count(End_points)
        #plt.imshow(End_points)
                
        Junction_points = cell_img[:, :, 1] == 255
        num_junction_points = count(Junction_points)
        #plt.imshow(Junction_points)
        
        Length = cell_img[:, :, 2]
        branches_length = count(Length) + num_end_points
        
                
        Branches = cell_img[:, :, 2]
        num_branches = count_branches(Branches)
        #plt.imshow(Branches)
        
        Initial_points = cell_img[:,:,1] == 200
        # Count the initial points
        num_initial_points = count(Initial_points)
        #plt.imshow(Initial_points)
               
        df_branches.loc[len(df_branches)] = [cell, num_end_points, num_junction_points, num_branches, num_initial_points, branches_length]

# Extract numerical part from the "Cell" column
df_branches['Cell'] = df_branches['Cell'].str.extract(r'(\d+)')      



####GENERANDO EL DATAFRAME UNIDO POR LAS CARACTERISTICAS DE LAS RAMIFICACIONES: 
    #función 

merged_df = df_area.merge(df_positions, on="Cell", how="inner")

 

merged_df = df_branches.merge(df_area, on="Cell", how="inner")


print ("...")
print ("...")
print("Generating DATAFRAME ...")
print ("...")
print ("...")

#MODIFICAR ACORDE A TU SUJETO. ESTAS COLUMNAS SE AGREGAN MANUALMENTE AL DATAFRAME


subject = "sub-001"
group = "mor"
region = "Crus"
side = "left"
slice_number = "A"


############################################## Add columns 


# Sort the DataFrame by the "Cell" column as numeric values
merged_df = merged_df.sort_values(by='Cell', key=lambda x: x.astype(int))

# Reset the index after sorting if needed
merged_df = merged_df.reset_index(drop=True)




merged_df.insert(0, 'subject', subject)
merged_df.insert(1, 'group', group)
merged_df.insert(2, 'region', region)
merged_df.insert(3, 'side', side)
merged_df.insert(4, 'slice', slice_number)






############################################## FRACTAL ANALYSIS: GEOMETRY


# Calculate the ratio_area with the condition, and set to 0 when needed
#merged_df['ratio_area'] = np.where((merged_df['polygon_areas'] == 0) | (merged_df['Area_soma'] == 0), 0, merged_df['polygon_areas'] / merged_df['Area_soma'])




merged_df["centroid_cell"] = centroids_cell
merged_df["centroid_polygon"] = centroids_polygon
merged_df["cell_positions"] = cell_positions

merged_df['distance_centroids'] = merged_df.apply(lambda row: distance.euclidean(row['centroid_cell'], row['centroid_polygon']), axis=1)


merged_df["polygon_areas"] =polygon_areas
merged_df["polygon_perimeters"] =polygon_perimeters
merged_df["polygon_compactness"] =polygon_compactness 
#merged_df["soma_compactness"] =soma_compactness 


merged_df["eccentricities"]= eccentricities
merged_df["feret_diameters"]= feret_diameters
merged_df["orientations"]= orientations


# Save the results to a CSV file
#merged_df.to_csv(csv_path + "Cell_New_Morphology.csv", index=False)




############################################## Depurando: ELIMINANDO CELULAS MAL CAPTADAS:
#Skeleton_Soma y ID para identificar celulas mal captadas

values_to_remove = ["22", "23", "32", "36", "66", "72"]

# Use the ~ (tilde) operator to negate the condition and keep rows where 'Cell' is not in the list
merged_df = merged_df[~merged_df['Cell'].isin(values_to_remove)]




#SAVE DATAFRAME

merged_df.to_csv(csv_path + "Cell_Morphology.csv", index=False)


