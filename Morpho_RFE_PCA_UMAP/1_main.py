#!/usr/bin/env python3
# -*- coding: utf-8 -*-





import os
import cv2
import pandas as pd
import numpy as np
from morpho import set_path, save_tif, erase, count, count_branches, detect_and_color, gammaCorrection, calculate_area, find_contours, polygon, detect_features, name_to_number, sholl_circles
import tifffile as tiff
#from morpho2 import sholl_circles
#import tkinter as tk
#import PySimpleGUI as sg
from skimage import io, filters
#import matplotlib.pyplot as plt
#from tkinter import filedialog, Tk
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
#from skimage import restoration
#import re
from scipy.spatial import distance




#PATHS


i_path="/Users/juanpablomayaarteaga/Desktop/Hilus/Prepro/"
o_path= set_path(i_path+"/Output_images/")
ID_path= set_path(o_path+"ID/")


#VARIABLES

subject = ["R1", "R2"]
group = ["CNEURO1", "VEH"]
treatmeant = ["ESC", "SS"]
tissue = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"]



           
#For loop to process all the TIF files in the 

for s in subject:
    for g in group:
        for tr in treatmeant:
            for ti in tissue:
                 
                 # READING THE INDIVIDUAL TIF FILES OF THE DIRECTORY
                 
                 img = f"{s}_{g}_{tr}_{ti}_CA1_FOTO1_PROCESSED"
                 tif= ".tif"

                
                 image_path = i_path + img + tif
                
                
                 if os.path.isfile(image_path):
                     

                    # SETTING NEW PATHS (THEREFORE DIRECTORIES) TO STORE EACH ANALYSIS 
                    # AND SUPERVISED THAT ALL THE METRICS ARE WELL PROCESSED
                    
                    print ("...")
                    print ("...")
                    print("Setting paths...")
                    print ("...")
                    print ("...")
                    
                    
                    individual_img_path= set_path(o_path+f"{s}_{g}_{tr}_{ti}_CA1_FOTO1_PROCESSED/")
                    
                    Cells_path= set_path(individual_img_path+"Cells/")
                    
                    Cells_thresh_path= set_path(Cells_path+"Cells_thresh/")
                    
                    Soma_path= set_path(Cells_path+"Soma/")
                    
                    Skeleton_path= set_path(Cells_path+"Skeleton/")
                    
                    Branches_path= set_path(Cells_path+"Branches/")
                    
                    Skeleton2_path= set_path(Cells_path+"Skeleton2/")

                    Branches2_path= set_path(Cells_path+"Branches2/")
                    
                    Skeleton_Soma_path= set_path(Cells_path+"Skeleton_Soma/")
                    
                    Soma_centroid_path= set_path(Cells_path+"Soma_Centroids/")
                    
                    Polygon_path= set_path(Cells_path+"Polygon/")
                    
                    Polygon_Centroid_path= set_path(Cells_path+"Polygon_Centroid/")
                    
                    Branches3_path= set_path(Cells_path+"Branches3/")
                    
                    Cell_centroid_path= set_path(Cells_path+"Cell_centroid/")
                    
                    Sholl_path= set_path(Cells_path+"Sholl/")
                    
                    csv_path= set_path(individual_img_path+"Data/")
                    
                    
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    image = cv2.bitwise_not(image)
                    
                    #THE IMAGES ARE ALREADY PREPROCESSED AND THERSHOLDED
                    thresh = image

                    # LABEL OBJECTS
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=4)
                    
                    # SET PARAMETERS TO EXCLUDE NOISE 
                    min_area = 2000
                    max_area = 24000000
                    #max_area=6000
                    num_cells_filtered = 0
                   
              
                    #### 
                    print ("...")
                    print ("...")
                    print("Identifying individual cells ORIGINAL...")
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
                            output_filename = f"{img}_cell_{individual_cell}.tif"
                            output_path = os.path.join(Cells_thresh_path, output_filename)
                            tiff.imwrite(output_path, object_img)
                            
                            
                    
                    
                    # Save the threshold image with red rectangles around objects
                    #save_tif(images, name="_rectangles.tif", path=Label_path, variable=image)
                    print ("...")
                    print ("...")
                    print("Individual cells ORIGINAL identified")
                    print ("...")
                    print ("...")



###############################################################################
####################################################################################
###############################################################################################
####################################################################################################            
##########################################################################################
###############################################################################      
    

    
            # IDENTIFYING THE CELLS IN THE THERSHOLDED IMAGE
            # CREATING AN ID TO EACH CELL
                   
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
                    save_tif(output_filename, name=".tif", path=ID_path, variable=color_image)
            
            
                    print ("...")
                    print ("...")
                    print("Reference image ready!")
                    print ("...")
                    print ("...")
                    
                    
                    # CAPTURING THE POSITION OF THE CELLS IN ITS ORIGINAL IMAGE
                    name_to_number(Cells_thresh_path)
                    df_positions = pd.DataFrame({'Cell': cell_num, 'cell_positions': cell_positions})
                    
                    #CREATING A DF WITH THE CELL POSITIONS
                    # Sort the DataFrame by the "Cell" column as numeric values
                    df_positions = df_positions.sort_values(by='Cell', key=lambda x: x.astype(int))
                    df_positions.to_csv(csv_path + "Cell_Positions.csv", index=False)

                   

                    
                   

                
                    



###############################################################################################################
###############################################################################################################
###############################################################################################################
##############################################  REMOVE OBJECTS ##################################################




                    ### SUBSTACTING OBJECTS NOT CORRESPONDING TO THE CELLS 
                    # (BRANCESS FROM OTHER CELLS THAT WERE CLOSE TO THE CELL AND, THEREFORE
                    # CAPTURED BY THE BOX DELIMITATING THE CELL)
                    
                    print ("...")
                    print ("...")
                    print("Substracting OBJECTS NOT CORRESPONDING TO THE CELLS...")
                    print ("...")
                    print ("...")
                    
                    for images in os.listdir(Cells_thresh_path):
                        if images.endswith(".tif"):
                            
                            input_file = os.path.join(Cells_thresh_path, images)
                            image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)

                            # Find connected components and label them
                            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
                            # Find the label with the largest area (THE CELL)
                            largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1  # Exclude background label (0)
                            
                            # Create a mask to keep only the largest object
                            mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)
                            # Apply the mask to the original image to keep only the largest object
                            result = cv2.bitwise_and(image, image, mask=mask)
                            save_tif(images, name=".tif", path=Cells_thresh_path, variable=result)
                        
                        
                        
                        
                        
                        else:
                            continue
                            
                            


                    
                    print ("...")
                    print ("...")
                    print("NOISE REMOVED!")
                    print ("...")
                    print ("...")
                    
                    




##############################################  REMOVE OBJECTS ##################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################







###############################################################################################################
###############################################################################################################
###############################################################################################################
##############################################  CELL FEATURES ##################################################


                    # FIRST ANALYSIS: FEATURES OF THE CELL
                    # DF OF THRESHOLDED CELL FEATURES: AREA, PERIMETER, CIRCULARITY, ...
                                 
                    print ("...")
                    print ("...")
                    print("Substracting CELL FEATURES...")
                    print ("...")
                    print ("...")
                    
                    
                    df_cell = pd.DataFrame(columns=["Cell", "cell_area", "cell_perimeter", "cell_circularity", "cell_compactness", "cell_orientation", "cell_feret_diameter", "cell_eccentricity", "cell_aspect_ratio"])  # Initialize DataFrame with column names
                    
                    # Cell Area, Perimeter, and Circularity
                    for cell in os.listdir(Cells_thresh_path):
                        if cell.endswith(".tif"):
                            input_cell = os.path.join(Cells_thresh_path, cell)
                            cell_area = calculate_area(input_cell)
                            
                            if area != 0:
                                contours = find_contours(input_cell)  
                                
                                cell_contour = max(contours, key=cv2.contourArea)
                                
                                # Calculate the eccentricity
                                if len(cell_contour) >= 5:
                                    _, (major_axis, minor_axis), _ = cv2.fitEllipse(cell_contour)
                                    cell_eccentricity = major_axis / minor_axis
                                
                                
                                # Calculate the bounding box of the contour
                                x, y, width, height = cv2.boundingRect(cell_contour)
                                
                                # Calculate the aspect ratio
                                cell_aspect_ratio = width / float(height)
                                
                                
                                # Calculate the Feret diameter (maximum caliper)
                                min_rect = cv2.minAreaRect(cell_contour)
                                cell_feret_diameter = max(min_rect[1])  # Maximum of width and height
                                
                                # Calculate the orientation (angle of the Feret diameter)
                                cell_orientation = min_rect[2]  # Angle in degrees
                                
                                # Calculate the perimeter
                                cell_perimeter = 0
                                for contour in contours:
                                    cell_perimeter += cv2.arcLength(contour, closed=True) 
                                
                                # Calculate circularity and compactness only if the perimeter is non-zero
                                if cell_perimeter != 0:
                                    cell_circularity = 4 * np.pi * (cell_area / (cell_perimeter ** 2))
                                    cell_compactness = cell_area / cell_perimeter
                                    
                                
                            else:
                                cell_area = 0
                                cell_feret_diameter = 0
                                cell_orientation = 0
                                cell_perimeter = 0
                                cell_circularity = 0 
                                cell_compactness = 0
                                cell_eccentricity = 0
                                cell_aspect_ratio = 0
                                
                            
                            
                            df_cell.loc[len(df_cell)] = [cell, cell_area, cell_perimeter, cell_circularity, cell_compactness, cell_orientation, cell_feret_diameter, cell_eccentricity, cell_aspect_ratio]
                    
                    
                    
                    # Extract the numeric part from the "Cell" column and convert it to integers
                    df_cell['Cell'] = df_cell['Cell'].str.extract(r'(\d+)').astype(int)
                    # Sort the DataFrame by the "Cell" column in ascending order
                    df_cell = df_cell.sort_values(by='Cell')
                    #df_area['Cell'] = df_area['Cell'].astype(str)
                    df_cell.to_csv(csv_path + "Cell_features.csv", index=False)
                    
                    
                    
                    
                    
                    
                    print ("...")
                    print ("...")
                    print("CELL FEATURES substracted!")
                    print ("...")
                    print ("...")


##############################################  CELL FEATURES ##################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################




###############################################################################################################
###############################################################################################################
###############################################################################################################
##############################################  CELL SOMA ##################################################

                    ### SECOND ANALYSIS: SOMA OF THE CELL
                    # SUBSTRACTING THE SOMA OF THE CELL: CALCULATING ITS CIRCULARITY, AREA, PERIMETER, ...
         
            
                    # FIRST, THE SOMA IS SUBSTRACTED FROM THE THERSHOLDED CELL
                    # SECOND, THE FEATURES OF THE SOMA AREA ANLYZED
           
                    print ("...")
                    print ("...")
                    print("Substracting soma of the cells...")
                    print ("...")
                    print ("...")
                    
                    for images in os.listdir(Cells_thresh_path):
                        if images.endswith(".tif"):
                            input_file = os.path.join(Cells_thresh_path, images)
                            image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
                            
                            
                            # Opening preprocessing to dealing with the branches
                            kernel = np.ones((3,3),np.uint8)
                            
                            #REMOVING BRANCHES TO JUST ANALYSE THE SOMA: ERODE AND LARGEST OBJECT
                            eroded = cv2.erode(image, kernel, iterations=4)
                            # Find connected components and label them
                            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(eroded, connectivity=8)
                            # Find the label with the largest area
                            largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1  # Exclude background label (0)
                            # Create a mask to keep only the largest object
                            mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)
                            # Apply the mask to the original image to keep only the largest object
                            result = cv2.bitwise_and(eroded, eroded, mask=mask)
                            
                            #RESTORING THE ORIGINAL SIZE THAT WAS ERODE
                            dilated = cv2.dilate(result, kernel, iterations=4)
                            
                            # Get image dimensions
                            height, width = dilated.shape[:2]
                            
                            # Define the size of the frame
                            frame_thickness = 10
                            
                            # Create a mask to identify the frame area
                            mask = np.ones(dilated.shape[:2], dtype="uint8") * 255  # Initialize a white mask
                            
                            # Set the top row, bottom row, and both sides of the frame to black in the mask
                            mask[:frame_thickness, :] = 0  # First row at the top
                            mask[height - frame_thickness:, :] = 0  # Bottom row at the bottom
                            mask[:, :frame_thickness] = 0  # First column at the left
                            mask[:, width - frame_thickness:] = 0  # Last column at the right
                            
                            # Apply the mask to the image
                            result = cv2.bitwise_and(dilated, dilated, mask=mask)


      

                            save_tif(images, name=".tif", path=Soma_path, variable=result)
                        
                        else:
                            continue
                            
                            
                    name_to_number(Soma_path)

                    
                    print ("...")
                    print ("...")
                    print("Soma substracted!")
                    print ("...")
                    print ("...")
                    
                    

################################################# ANALSIS AREA Y PERIMETRO #################################################

                    print ("...")
                    print ("...")
                    print("ANALYZING Soma features...")
                    print ("...")
                    print ("...")

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
                    #df_area['Cell'] = df_area['Cell'].astype(str)
                    
                    
                    df_area.to_csv(csv_path + "Soma_features.csv", index=False)
                    
                    
                    
                    
                    
                    
                    print ("...")
                    print ("...")
                    print("Soma features ANALYZED!")
                    print ("...")
                    print ("...")



################################################# SOMA CELULAR #########################################
###############################################################################################################
###############################################################################################################






###############################################################################################################
###############################################################################################################
################################################# ANALSIS SKELETONIZE #########################################



# ANALISIS TRADICIONAL DE ZHAN DE SKELETONIZE QUE SE USA EN IMAGEJ PARA MORFOLOGIA CON MEJORAS
                    
                    ### THIRD ANALYSIS: BRANCHES OF THE CELL
                    # SUBSTRACTING THE BRANCHES OF THE CELL: CALCULATING ITS END POINTS, JUNCTIONS POINTS, ...
           
              
                    # FIRST, THE CELL IS THINNING FROM THE THERSHOLDED CELL
                    # TO OBTAIN THE SKELETON OF THE CELL
                    # SECOND, THE FEATURES OF THE SKELETON AREA ANLYZED
                    # THE REGION CORRESPONDING TO THE SOMA IS SUBSTRACTED
                    # TO ENSURE THE ONLY ANALYSIS OF THE BRANCHES AND TO OBTAIN 
                    # THE CELL PROCESS (HERE DENOMINATED "INITIAL POINTS")
                

    
                    print ("...")
                    print ("...")
                    print("Substracting Cell Ramifications features...")
                    print ("...")
                    print ("...")
                    
                    
                    #SE ESTABLECE QUE COLUMNAS TENDRA NUESTRO DATAFRAME 
                    df_skeletonze= pd.DataFrame(columns=["Cell", "End_Points", "Junctions", "Branches", "Total_Branches_Length"])
                    
                    
                    #Skeletonize and detect branches
                    
                    for cell in os.listdir(Cells_thresh_path):
                        if cell.endswith(".tif"):
                            input_cell = os.path.join(Cells_thresh_path, cell)
                            cell_img = cv2.imread(input_cell, cv2.IMREAD_GRAYSCALE)
                            scale = cell_img /255
                            
                            #Skeletonize
                            skeleton = skeletonize(scale)
                            clean_skeleton= erase(skeleton, 40)
                            save_tif(cell, name=".tif", path=Skeleton_path, variable=clean_skeleton)
                            
                            #Detecteding Branches features
                            M, colored_image= detect_and_color(clean_skeleton)
                            save_tif(cell, name=".tif", path=Branches_path, variable=colored_image)
                     
                        
                    name_to_number(Branches_path)         
                      
                    # IDENTIFYING THE FEATURES OF THE SKELETON: END POINTS, JUNCTIONS POINTS, ...
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
                                    
                            df_skeletonze.loc[len(df_skeletonze)]=[cell, num_end_points, num_junction_points, num_branches, branches_length ]
                            
                    # SKELETON DF
                    df_skeletonze['Cell'] = df_skeletonze['Cell'].str.extract(r'(\d+)')   
                    df_skeletonze = df_skeletonze.sort_values(by='Cell')
                    #df_skeletonze['Cell'] = df_skeletonze['Cell'].astype(str)    
                    df_skeletonze.to_csv(csv_path+"Skeletonize.csv", index=False)     
                    
                    name_to_number(Skeleton_path)
                    
                    
                    
                    print ("...")
                    print ("...")
                    print("Cell Ramifications features substracted!")
                    print ("...")
                    print ("...")




################################################# ANALSIS SKELETONIZE #########################################
###############################################################################################################
###############################################################################################################
###############################################################################################################



###############################################################################################################
###############################################################################################################
###############################################################################################################
################################################# SKELETON WITHOUT SOMA ##########################################



                    print ("...")
                    print ("...")
                    print("Getting the Skeleton whitout Soma")
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
                                save_tif(image, ".tif", Skeleton2_path, subtracted_image)
                    
                                # Detecting branches features
                                M, colored_image = detect_and_color(subtracted_image)
                                save_tif(image, name=".tif", path=Branches2_path, variable=colored_image)
                    
                                added_image = cv2.add(skeleton_img, soma_img)
                                save_tif(image, ".tif", Skeleton_Soma_path, added_image)
                    
                     
                    
                    
                    print ("...")
                    print ("...")
                    print("SKELETON substracted!")
                    print ("...")
                    print ("...")



################################################# ESQUELETO SIN SOMA ##########################################                  
###############################################################################################################
###############################################################################################################
###############################################################################################################




###############################################################################################################
###############################################################################################################
###############################################################################################################
################################################# SHOLL ANALYSIS ###################################################


                    
                    print ("...")
                    print ("...")
                    print("Getting Cell centroids...")
                    print ("...")
                    print ("...")
                    
                    
                    centroids_soma = []
                    #df_sholl= pd.DataFrame(columns=["Cell", "centroids_soma", "sholl_max_dist", "sholl_circles", "sholl_crossing_points"])
                    for images in os.listdir(Soma_path):
                        if images.endswith(".tif"):
                            input_file = os.path.join(Soma_path, images)
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
                                    centroids_soma.append((centroid_x, centroid_y))
                                    
                            
                                    print(f"Centroid coordinates: {centroid_x}, {centroid_y}")
                                else:
                                    print("Object has no area (m00=0)")
                            else:
                                print("No contours found in the image")
                                
                            img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                            
                            # Optionally, you can draw the centroid on the image for visualization
                            centroid_img= cv2.circle(img_color, centroid, 1, (255, 0, 0), -1)  # Draw a red circle at the centroid
                            save_tif(images, name=".tif", path=Soma_centroid_path, variable=centroid_img)
                            #plt.imshow(centroid_img)
                    
                    
                    print ("...")
                    print ("...")
                    print("Cell centroids added!!")
                    print ("...")
                    print ("...")
                    
                    
                    
                    # FORTH ANALYSIS: SHOLL ANALYSIS
                    # GENERATING THE SKELETON WITH THE SOMA TO ASSES THE CROSSING PROCESSES, ETC
  
                    
                    print ("...")
                    print ("...")
                    print("Generating THE IMAGES FOR SHOLL ANALYSIS...")
                    print ("...")
                    print ("...")
                    
                    
                    
                    for image in os.listdir(Skeleton_path):
                        if image.endswith(".tif"):
                            input_skeleton = os.path.join(Skeleton_path, image)
                            input_soma = os.path.join(Soma_centroid_path, image)
                            
                            
                            skeleton_img = cv2.imread(input_skeleton, cv2.IMREAD_COLOR)
                            soma_img = cv2.imread(input_soma, cv2.IMREAD_COLOR)
                            soma_img = cv2.cvtColor(soma_img, cv2.COLOR_BGR2RGB)
                            

                            
                            # Check if the images have the same dimensions
                            if skeleton_img.shape == soma_img.shape:
                                cell_centroid_image = cv2.add(skeleton_img, soma_img)
                                save_tif(image, ".tif", Cell_centroid_path, cell_centroid_image)
                    
                    
                    
                    print ("...")
                    print ("...")
                    print("SHOLL ANALYSIS...")
                    print ("...")
                    print ("...")
                    
                    # SHOLL ANALYSIS
                    
                    df_sholl= pd.DataFrame(columns=["Cell", "sholl_max_distance", "sholl_crossing_processes", "sholl_num_circles"])
                    
                    for image in os.listdir(Cell_centroid_path):
                        if image.endswith(".tif"):
                            input_sholl = os.path.join(Cell_centroid_path, image)
                            #input_soma = os.path.join(Soma_path, image)
                            
                            sholl_img = cv2.imread(input_sholl, cv2.IMREAD_COLOR)
                            #soma = cv2.imread(input_soma)
                            
                            
                            #sholl_img = cv2.imread(input_sholl, cv2.IMREAD_COLOR)
                            #sholl_image, sholl_max_distance, sholl_crossing_processes, circle_image = sholl_circles(sholl_img, soma)
                            sholl_image, sholl_max_distance, sholl_crossing_processes, circle_image = sholl_circles(sholl_img)

                            circles = circle_image[:, :, 2]
                            sholl_num_circles = count(circles)

                            save_tif(image, ".tif", Sholl_path, sholl_image)
                            
                            
                            
                            
                            df_sholl.loc[len(df_sholl)] = [image, sholl_max_distance, sholl_crossing_processes, sholl_num_circles]
                    
                    
                    df_sholl['Cell'] = df_sholl['Cell'].str.extract(r'(\d+)')   
                    df_sholl = df_sholl.sort_values(by='Cell')   
                    df_sholl.to_csv(csv_path+"Sholl_Analysis.csv", index=False) 
                    
                    
                    print ("...")
                    print ("...")
                    print("SHOLL ANALYSIS DONE!")
                    print ("...")
                    print ("...")



################################################# SHOLL ANALYSIS ###################################################
###############################################################################################################
###############################################################################################################
############################################################################################################### 

                    
     
###############################################################################################################
###############################################################################################################
###############################################################################################################
################################################# GEOMETRIA: FRACTAL ANALYSIS ###################################################
                   
                    
                    # FIFTH ANALYSIS: FRACTAL ANALYSIS
                    #POLYGON FROM THE THRESHOLDED CELL

                    print ("...")
                    print ("...")
                    print("Generating POLYGONS...")
                    print ("...")
                    print ("...")
                    
                            
                    
                    polygon_areas = []
                    polygon_perimeters = []
                    polygon_compactness = []
                    
                    
                    df_polygon= pd.DataFrame(columns=["Cell", "polygon_area", "polygon_perimeters", "polygon_compactness"])
                    for image in os.listdir(Cells_thresh_path):
                        if image.endswith(".tif"):
                            input_file = os.path.join(Cells_thresh_path, image)
                            img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
                            
                            contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    
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
                            save_tif(image, ".tif", Polygon_path, polygon_image)
                            
                            df_polygon.loc[len(df_polygon)]=[image, area,perimeter, compactness ]

 
                    

                    
                    
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
                    
                    
                    polygon_eccentricities = []
                    polygon_feret_diameters = []
                    polygon_orientations = []
                    
                    
                    df_polygon2= pd.DataFrame(columns=["Cell", "polygon_eccentricities", "polygon_feret_diameters", "polygon_orientations"])

                    for image in os.listdir(Polygon_path):
                        if image.endswith(".tif"): 
                            input_polygon = os.path.join(Polygon_path, image)
                            polygon_img = io.imread(input_polygon)    
                            
                            contours, _ = cv2.findContours(polygon_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                            if len(contours) > 0:
                                polygon_contour = max(contours, key=cv2.contourArea)  # Obtain the largest contour

                                # Calculate the eccentricity
                                _, (major_axis, minor_axis), _ = cv2.fitEllipse(polygon_contour)
                                eccentricity = major_axis / minor_axis
                                polygon_eccentricities.append(eccentricity)
                            
                                # Calculate the Feret diameter (maximum caliper)
                                min_rect = cv2.minAreaRect(polygon_contour)
                                feret_diameter = max(min_rect[1])  # Maximum of width and height
                                polygon_feret_diameters.append(feret_diameter)
                            
                                # Calculate the orientation (angle of the Feret diameter)
                                orientation = min_rect[2]  # Angle in degrees
                                polygon_orientations.append(orientation)
                    

                                centroid_x = int(M['m10'] / M['m00'])
                                centroid_y = int(M['m01'] / M['m00'])
                                centroid = (centroid_x, centroid_y)
                                # ... (Previous code)
                    
                                # Calculate the eccentricity
                                _, (major_axis, minor_axis), _ = cv2.fitEllipse(polygon_contour)
                                eccentricity = major_axis / minor_axis
                                polygon_eccentricities.append(eccentricity)
                    
                                # Calculate the Feret diameter (maximum caliper)
                                min_rect = cv2.minAreaRect(polygon_contour)
                                feret_diameter = max(min_rect[1])  # Maximum of width and height
                                polygon_feret_diameters.append(feret_diameter)
                    
                                # Calculate the orientation (angle of the Feret diameter)
                                orientation = min_rect[2]  # Angle in degrees
                                polygon_orientations.append(orientation)
                                
                                
                                df_polygon2.loc[len(df_polygon2)]=[image, eccentricity, feret_diameter, orientation ]
                                
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




                    df_polygon['Cell'] = df_polygon['Cell'].str.extract(r'(\d+)')   
                    
                    df_polygon = df_polygon.sort_values(by='Cell')
                    df_polygon2['Cell'] = df_polygon['Cell'].str.extract(r'(\d+)')   
                    df_polygon2 = df_polygon2.sort_values(by='Cell')
                    
                    
                    df_polygon = pd.merge(df_polygon, df_polygon2, on='Cell', how='inner')
                    
                    df_polygon.to_csv(csv_path+"Fractal_Analysis.csv", index=False) 
                    
                    print ("...")
                    print ("...")
                    print("Polygon centroids added!")
                    print ("...")
                    print ("...")
     
                    
     
        
     
################################################# GEOMETRIA: FRACTAL ANALYSIS ###################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################                    
   









###############################################################################################################
###############################################################################################################
###############################################################################################################
################################################# NEW SKELETON ANALYSIS #####################################
                 
                    
                    print ("...")
                    print ("...")
                    print("Generating INITIAL POINTS and Skeleton analysis...")
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
                                save_tif(image, name=".tif", path=Branches3_path, variable=colored_image3)
                            
                    
                    
                    df_branches = pd.DataFrame(columns=["Cell", "End_Points", "Junctions", "Branches", "Initial_Points", "Total_Branches_Length"])  # Initialize DataFrame with column names
                    
                    
                    
                    print ("...")
                    print ("...")
                    print("New Skeletonize procees done...")
                    print ("...")
                    print ("...")


                    # ANALYZING BRANCHES FEATURES
 
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
                    
  
                    
  
                    #df_branches.loc[df_branches["Initial_Points"] == 0, "Initial_Points"] += 1
                    # Calculate 'ratio_branches'
                    df_branches['ratio_branches'] = df_branches['End_Points'] / df_branches['Initial_Points']
                    column_name = 'ratio_branches'
                    
                    #column_index = 5
                    #df_branches = pd.concat([df_branches.iloc[:, :column_index], df_branches[column_name], df_branches.iloc[:, column_index:]], axis=1)
                         
                    
                    print ("...")
                    print ("...")
                    print("BRANCHES features detected")
                    print ("...")
                    print ("...")




################################################# NEW SKELETON ANALYSIS #####################################
###############################################################################################################
###############################################################################################################
###############################################################################################################






###############################################################################################################
###############################################################################################################
###############################################################################################################
################################################# DATAFRAME GENERAL #####################################



                    print ("...")
                    print ("...")
                    print("Generating DATAFRAME WITH ALL THE FEATURES ...")
                    print ("...")
                    print ("...")
                    
                    
                    
                    ####GENERANDO EL DATAFRAME UNIDO POR LAS CARACTERISTICAS DE LAS RAMIFICACIONES: 
                    df_positions['Cell'] = df_positions['Cell'].astype(int)
                    df_cell['Cell'] = df_cell['Cell'].astype(int)
                    df_area['Cell'] = df_area['Cell'].astype(int)
                    df_branches['Cell'] = df_branches['Cell'].astype(int)
                    df_polygon['Cell'] = df_polygon['Cell'].astype(int)
                    df_sholl['Cell'] = df_sholl['Cell'].astype(int)

                    merged_df = df_positions.merge(df_area, on="Cell", how="inner").merge(df_branches, on="Cell", how="inner").merge(df_polygon, on="Cell", how="inner").merge(df_cell, on="Cell", how="inner").merge(df_sholl, on="Cell", how="inner")
                    
                    # Assuming the merged dataframe is named 'merged_df'

                    merged_df['cell_solidity'] = merged_df['cell_area'] / merged_df['polygon_area']
                    merged_df['cell_convexity'] = merged_df['cell_perimeter'] / merged_df['polygon_perimeters']



                    #MODIFICAR ACORDE A TU SUJETO. ESTAS COLUMNAS SE AGREGAN MANUALMENTE AL DATAFRAME
                    #########    SUJETO, GRUPO, REGION ANALIZADA, LADO, CORTE 
                    
                    ############################################## Add columns 
                    
                    
                    # Sort the DataFrame by the "Cell" column as numeric values
                    merged_df = merged_df.sort_values(by='Cell', key=lambda x: x.astype(int))
                    
                    # Reset the index after sorting if needed
                    merged_df = merged_df.reset_index(drop=True)
                    
                    #THIS SECTION DEPENDS ON THE REGION AND SUBJECTS THAT YOU ARE ANALYZING
                    # THESE FEATURES COULD BE EXTRACTED FROM THE FILE BUT TO AVOID MANIPULATING
                    # THE NAME OF THE FILES, THESE COLUMNS ARE MANUALLY CREATED
                    
                    #REGION OF INTEREST
                    ca1="CA1"
                    
                    #GENERAL ID 
                    ID= f"{s}_{g}_{tr}_{ti}_{ca1}"
                    
                    # COLUMNS TO IDENTIFY THE SUBJECT
                    merged_df.insert(0, 'subject', s)
                    merged_df.insert(1, 'group', g)
                    merged_df.insert(2, 'treatment', tr)
                    merged_df.insert(3, 'tissue', ti)
                    merged_df.insert(4, 'region', ca1)
                    

                    merged_df['ID'] = (merged_df['subject'] + "_" + merged_df['group'] + "_" +
                                       merged_df['treatment'] + "_" + merged_df['tissue'] + "_" +
                                       merged_df['region'])
                    
                    
                    merged_df = merged_df[ ['ID'] + [col for col in merged_df.columns if col != 'ID'] ]

                    
                    #SAVE DATAFRAME
                    csv_name = csv_path + "Cell_Morphology.csv"
                    merged_df.to_csv(csv_name, index=False)


                    print ("...")
                    print ("...")
                    print(f"DATAFRAME SAVED AS CSV AT: {csv_name}")
                    print ("...")
                    print ("...")



