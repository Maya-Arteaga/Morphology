#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 19:11:01 2023

@author: juanpablomayaarteaga
"""


import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from morpho import set_path, gammaCorrection


n_neighbors=15


i_path = "/Users/juanpablomayaarteaga/Desktop/Hilus/Prepro/"
o_path = os.path.join(i_path, "Output_images/")
ID_path= set_path(o_path+f"ID_clusters_PCA_UMAP_{n_neighbors}/")

i_original_path = "/Users/juanpablomayaarteaga/Desktop/Hilus/"




subject = ["R1", "R2"]
group = ["CNEURO1", "VEH"]
treatment = ["ESC", "SS"]
tissue = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"]

df_color_position = pd.DataFrame(columns=["Cell", "Cluster_Labels", "x1", "y1", "x2", "y2"])

for s in subject:
    for g in group:
        for tr in treatment:
            for ti in tissue:
                
                individual_img_path = os.path.join(o_path, f"{s}_{g}_{tr}_{ti}_CA1_FOTO1_PROCESSED/")
                csv_path = os.path.join(individual_img_path, "Data/")
                csv_file = f"{s}_{g}_{tr}_{ti}_Morphology_PCA_UMAP_HDBSCAN_{n_neighbors}.csv"
                original_img = f"{s}_{g}_{tr}_{ti}_CA1_FOTO1_PROCESSED.tif"
                

                if os.path.isfile(os.path.join(csv_path, csv_file)):
                    data = pd.read_csv(os.path.join(csv_path, csv_file))
                    
                    if os.path.exists(individual_img_path):

                        Cells_path = os.path.join(individual_img_path + "Cells/")
                        color_path = set_path(Cells_path + "Color_Cells/")
                        Cells_thresh_path = set_path(Cells_path + "Cells_thresh/")
    
                        if os.path.exists(Cells_thresh_path):
                            for _, row in data.iterrows():
                                cell_number = row['Cell']
                                cluster = row['Cluster_Labels']
                                coordinates = eval(row['cell_positions'])
                                x1, y1, x2, y2 = coordinates
    
                                # Construct the image filename based on cell_number
                                image_filename = f"{cell_number}.tif"
    
                                if os.path.exists(Cells_thresh_path) and image_filename in os.listdir(Cells_thresh_path):
                                    input_image = os.path.join(Cells_thresh_path, image_filename)
    
                                    if os.path.isfile(input_image):
                                        # Read the grayscale image
                                        image = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
                                        # Continue with image processing...
    
        
                                    if image_filename in os.listdir(Cells_thresh_path):
                                        input_image = os.path.join(Cells_thresh_path, image_filename)
        
                                        if os.path.isfile(input_image):
                                            # Read the grayscale image
                                            image = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
                                            # Continue with image processing...
        
                                            
        
                                        
                                            """
                                            bright_img = cv2.convertScaleAbs(image, alpha=3, beta=0)
                                            
                                            gaussian_array = filters.gaussian(bright_img, sigma=2.2)
                                            gaussian_image = (gaussian_array * 255).astype(np.uint8)
                                            gammaImg = gammaCorrection(gaussian_image, 1.25)
                                            """
                                            
                                            # Apply thresholding
                                            threshold_value = 50  # Adjust this value as needed
                                            _, thresholded_mask = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
                                            
                                            """
                                            #Colors according to plot  BGR format
                                            cluster_colors = {
                                                #4: (200, 200, 0),   # Purple/Yellow
                                                1: (200, 200, 0),   # Cyan
                                                3: (200, 0, 200),   # Magenta
                                                2: (0, 200, 200),
                                                0: (0, 0, 255), 
                                                5: (255, 255, 255),
                                                4: (0, 200, 0),     # Red
                                                #4: (200, 200, 0),     # Green
                                                #5: (0, 255, 0),     # Ligth Orange
                                                #6: (255, 255, 255), # White
                                                #7: (128, 128, 0),   # Light Orange
                                                #8: (128, 128, 0),   # Olive
                                                #9: (128, 0, 0),     # Maroon
                                                #10: (0, 0, 255),    # Blue
                                                #11: (0, 128, 0)     # Dark Green
                                            }
                                            """
                                            #Colors according to plot  BGR format
                                            cluster_colors = {
                                                3: (0, 0, 255),
                                                1: (200, 200, 0),   # 
                                                0: (0, 200, 0),
                                                2: (200, 0, 200),   # 
                                                4: (255, 255, 255),    #
                                                5: (0, 0, 125),
                                                -1: (125, 125, 125)
                                            }
                                            
                                            
                                        
                                            # Assign a color depending on its cluster number
                                            if cluster in cluster_colors:
                                                color = cluster_colors[cluster]
                                            else:
                                                color = (0, 0, 0)  # Default color if cluster is not found
                                
                                            
                                            colored_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                                
                                            # Apply the color to the thresholded regions using the mask
                                            colored_image[thresholded_mask == 255] = color
                                            
                                            # Convert BGR to BGRA and set the alpha channel for black background to 0
                                            rgba_image = cv2.cvtColor(colored_image, cv2.COLOR_BGR2BGRA)
                                            black_background = (colored_image[:, :, 0] == 0) & (colored_image[:, :, 1] == 0) & (colored_image[:, :, 2] == 0)
                                            rgba_image[:, :, 3][black_background] = 0
                                
                                            # Save the colored thresholded image with a unique filename
                                            output_image_path = os.path.join(color_path, f"{cell_number}.tif")
                                            cv2.imwrite(output_image_path, rgba_image)
                                            
                                
                                            # Add the cell information to the DataFrame
                                            df_color_position.loc[len(df_color_position)] = [f"{cell_number}.tif", cluster, x1, y1, x2, y2]
                                            
                                            




df_color_position = pd.DataFrame(columns=["Cell", "Cluster_Labels", "x1", "y1", "x2", "y2"])

for s in subject:
    for g in group:
        for tr in treatment:
            for ti in tissue:
                original_img = f"{s}_{g}_{tr}_{ti}_CA1_FOTO1_PROCESSED.tif"
                individual_img_path = os.path.join(o_path, f"{s}_{g}_{tr}_{ti}_CA1_FOTO1_PROCESSED/")
                csv_path = os.path.join(individual_img_path, "Data/")
                csv_file = f"{s}_{g}_{tr}_{ti}_Morphology_PCA_UMAP_HDBSCAN_{n_neighbors}.csv"

                if os.path.isfile(os.path.join(i_path, original_img)):
                    original_image = cv2.imread(os.path.join(i_path, original_img))
                    if original_image is not None:

                        original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2RGBA)
                        height, width, channels = original_image.shape
                        empty_image = np.zeros((height, width, 3), np.uint8)
                        
                        #empty_image = np.zeros((height, width, 4), np.uint8)
                        #empty_image[:, :, 3] = 255

                        
                        if os.path.isfile(os.path.join(csv_path, csv_file)):
                            data = pd.read_csv(os.path.join(csv_path, csv_file))
                            
                            Cells_path = os.path.join(individual_img_path + "Cells/")
                            color_path = set_path(Cells_path + "Color_Cells/")

                            # Loop through the rows of the 'data' DataFrame
                            for _, row in data.iterrows():
                                cell_number = row['Cell']
                                cluster = row['Cluster_Labels']
                                coordinates = eval(row['cell_positions'])
                                x1, y1, x2, y2 = coordinates
                                
                                # Construct the image filename based on cell_number
                                image_filename = f"{cell_number}.tif"
                                
                                # Check if the image file exists in the directory
                                if image_filename in os.listdir(color_path):
                                    input_image = os.path.join(color_path, image_filename)
                                    
                                    # Check if the input image file exists
                                    if os.path.isfile(input_image):
                                        
                                        
                                        """
                                        # Read the colored image
                                        colored_image = cv2.imread(input_image, cv2.IMREAD_UNCHANGED)
                                        #empty_image = np.zeros((height, width, 4), np.uint8)
                                        #empty_image[:, :, 3] = 255
                                        colored_image = colored_image[:, :, :3]  # Retain only the RGB channels
                                        
                                        # Create a mask for the black background
                                        black_background_mask = (colored_image[:, :, 0] == 0) & (colored_image[:, :, 1] == 0) & (colored_image[:, :, 2] == 0)
                                        
                                        # Convert the colored image to RGBA
                                        rgba_colored_image = cv2.cvtColor(colored_image, cv2.COLOR_BGR2BGRA)
                                        
                                        # Set the alpha channel based on the black background mask
                                        rgba_colored_image[:, :, 3][black_background_mask] = 0
                                        
                                        # Calculate the dimensions of the region where the image will be pasted
                                        overlay_width = x2 - x1
                                        overlay_height = y2 - y1
                                        
                                        # Resize the colored image to fit the overlay region
                                        colored_image_resized = cv2.resize(rgba_colored_image, (overlay_width, overlay_height))
                                        
                                        # Overlay the resized colored image onto the empty image at the specified coordinates
                                        empty_image[y1:y2, x1:x2] = colored_image_resized

                                        
                                        
                                        
                                        
                                        """
                                        # Read the colored image
                                        colored_image = cv2.imread(input_image, cv2.IMREAD_UNCHANGED)
                                        colored_image = colored_image[:, :, :3]
                                        #rgba_image = cv2.cvtColor(colored_image, cv2.COLOR_BGR2BGRA)
                                        #black_background = (colored_image[:, :, 0] == 0) & (colored_image[:, :, 1] == 0) & (colored_image[:, :, 2] == 0)
                                        #rgba_image[:, :, 3][black_background] = 0
                                        #plt.imshow(rgba_image)
                                        
                                        # Calculate the dimensions of the region where the image will be pasted
                                        overlay_width = x2 - x1
                                        overlay_height = y2 - y1
                                        
                                        # Resize the colored image to fit the overlay region
                                        colored_image_resized = cv2.resize(colored_image, (overlay_width, overlay_height))
                                        
                                        # Overlay the resized colored image onto the empty image at the specified coordinates
                                        #empty_image[y1:y2, x1:x2] = colored_image_resized
                                        
                                        # Ensure the colored image is resized to fit the area you want to overlay
                                        colored_image_resized = cv2.resize(colored_image_resized, (x2 - x1, y2 - y1))
                                        
                                        # Overlay the images using alpha blending
                                        alpha = 0.3  # Adjust the alpha value as needed for transparency
                                        
                                        # Create a region of interest (ROI) on the empty image for overlay
                                        roi = empty_image[y1:y1 + colored_image_resized.shape[0], x1:x1 + colored_image_resized.shape[1]]
                                        
                                        # Perform the overlay using alpha blending
                                        cv2.addWeighted(roi, alpha, colored_image_resized, 1 - alpha, 0, roi)
                                        
                                        
                                        
                                        # Draw a yellow rectangle around the pasted image
                                        cv2.rectangle(empty_image, (x1, y1), (x2, y2), (0, 255, 255, 0), 2)
                                        
                                        # Write the cell number at the top of the box in red
                                        font = cv2.FONT_HERSHEY_SIMPLEX
                                        font_scale = 0.5
                                        font_color = (0, 0, 255)
                                        font_thickness = 2
                                        text_size = cv2.getTextSize(str(cell_number), font, font_scale, font_thickness)[0]
                                        text_x = x1 + (overlay_width - text_size[0]) // 2
                                        text_y = y1 - 5
                                        cv2.putText(empty_image, str(cell_number), (text_x, text_y), font, font_scale, font_color, font_thickness)
                            
                            # Save the resulting image
                            cluster_image_path = os.path.join(ID_path, f"{s}_{g}_{tr}_{ti}_CA1_Clustered_PCA_UMAP.tif")
                            cv2.imwrite(cluster_image_path, empty_image)

                                    
    
    
    
    
    
    
                                
