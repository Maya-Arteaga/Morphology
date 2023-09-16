#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 01:42:44 2023

@author: juanpablomayaarteaga
"""

import cv2
import os
import numpy as np
import pandas as pd
from morpho import set_path, gammaCorrection
import matplotlib.pyplot as plt
from skimage import filters

i_path="/Users/juanpablomayaarteaga/Desktop/Prueba_Morfo/"

o_path= set_path(i_path+"/Output_images/")

ID_path= set_path(o_path+"ID/")


Cells_path= set_path(o_path+"Cells/")

Cells_original_path= set_path(Cells_path+"Cells_original/")

color_path= set_path(Cells_path + "Color_Cells/")

cluster_path= set_path(o_path + "Cluster_image/")


csv_path= set_path(o_path+"Data/")


Plot_path= set_path(o_path+"Plots/")



# Load the data from the CSV file
data = pd.read_csv(csv_path + "Morphology_Scaled_8.csv")





# Create an empty DataFrame to store cell positions and cluster information
df_color_position = pd.DataFrame(columns=["Cell", "Cluster", "x1", "y1", "x2", "y2"])

# Loop through the rows of the original DataFrame 'data'
for _, row in data.iterrows():
    cell_number = row['Cell']
    cluster = row['Cluster']
    coordinates = eval(row['cell_positions'])
    x1, y1, x2, y2 = coordinates

    # Construct the image filename based on cell_number
    image_filename = f"{cell_number}.tif"

    # Check if the image file exists in the directory
    if image_filename in os.listdir(Cells_original_path):
        input_image = os.path.join(Cells_original_path, image_filename)

        # Check if the input image file exists
        if os.path.isfile(input_image):
            # Read the grayscale image
            image = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
            
            """
            bright_img = cv2.convertScaleAbs(image, alpha=3, beta=0)
            
            gaussian_array = filters.gaussian(bright_img, sigma=2.2)
            gaussian_image = (gaussian_array * 255).astype(np.uint8)
            gammaImg = gammaCorrection(gaussian_image, 1.25)
            """
            
            # Apply thresholding
            threshold_value = 50  # Adjust this value as needed
            _, thresholded_mask = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
            
            
            #Colors according to plot
            cluster_colors = {
                0: (0, 255, 255),   # Yellow
                1: (220, 255, 0),   # Cyan
                2: (255, 0, 255),   # Magenta
                3: (0, 0, 255),     # Red
                4: (128, 128, 0),   # Olive
                5: (0, 255, 0),     # Green
                6: (255, 255, 255), # White
                7: (0, 150, 255),   # Light Orange
                #8: (128, 128, 0),   # Olive
                9: (128, 0, 0),     # Maroon
                10: (0, 0, 255),    # Blue
                11: (0, 128, 0)     # Dark Green
            }



            
        
            # Assign a color depending on its cluster number
            if cluster in cluster_colors:
                color = cluster_colors[cluster]
            else:
                color = (0, 0, 0)  # Default color if cluster is not found

            # Convert the grayscale image to a 3-channel color image
            colored_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            # Apply the color to the thresholded regions using the mask
            colored_image[thresholded_mask == 255] = color

            # Save the colored thresholded image with a unique filename
            output_image_path = os.path.join(color_path, f"{cell_number}.tif")
            cv2.imwrite(output_image_path, colored_image)
            

            # Add the cell information to the DataFrame
            df_color_position.loc[len(df_color_position)] = [f"{cell_number}.tif", cluster, x1, y1, x2, y2]






#########################          EMPTY IMAGE            ##########################
# Create an empty image of the same size as the grayscale image
original_image = cv2.imread(i_path + "VTA.tif", cv2.IMREAD_GRAYSCALE)
plt.imshow(original_image, cmap='gray')


empty_image = np.zeros_like(original_image)
empty_image = cv2.cvtColor(empty_image, cv2.COLOR_GRAY2BGR)
plt.imshow(empty_image, cmap='gray')



# Loop through the rows of the 'data' DataFrame
for _, row in data.iterrows():
    cell_number = row['Cell']
    cluster = row['Cluster']
    coordinates = eval(row['cell_positions'])
    x1, y1, x2, y2 = coordinates
    
    # Construct the image filename based on cell_number
    image_filename = f"{cell_number}.tif"
    
    # Check if the image file exists in the directory
    if image_filename in os.listdir(color_path):
        input_image = os.path.join(color_path, image_filename)
        
        # Check if the input image file exists
        if os.path.isfile(input_image):
            # Read the colored image
            colored_image = cv2.imread(input_image)
            
            # Calculate the dimensions of the region where the image will be pasted
            overlay_width = x2 - x1
            overlay_height = y2 - y1
            
            # Resize the colored image to fit the overlay region
            colored_image_resized = cv2.resize(colored_image, (overlay_width, overlay_height))
            
            # Overlay the resized colored image onto the empty image at the specified coordinates
            empty_image[y1:y2, x1:x2] = colored_image_resized

# Save the resulting image
cluster_image_path = os.path.join(cluster_path, "Clustered_image.jpg")
cv2.imwrite(cluster_image_path, empty_image)











#################### BOXES #######################


# Create an empty image of the same size as the grayscale image
original_image = cv2.imread(i_path + "VTA.tif")
plt.imshow(original_image)

empty_image = np.zeros_like(original_image)
plt.imshow(empty_image)


# Loop through the rows of the 'data' DataFrame
for _, row in data.iterrows():
    cell_number = row['Cell']
    cluster = row['Cluster']
    coordinates = eval(row['cell_positions'])
    x1, y1, x2, y2 = coordinates
    
    # Construct the image filename based on cell_number
    image_filename = f"{cell_number}.tif"
    
    # Check if the image file exists in the directory
    if image_filename in os.listdir(color_path):
        input_image = os.path.join(color_path, image_filename)
        
        # Check if the input image file exists
        if os.path.isfile(input_image):
            # Read the colored image
            colored_image = cv2.imread(input_image)
            
            # Calculate the dimensions of the region where the image will be pasted
            overlay_width = x2 - x1
            overlay_height = y2 - y1
            
            # Resize the colored image to fit the overlay region
            colored_image_resized = cv2.resize(colored_image, (overlay_width, overlay_height))
            
            # Overlay the resized colored image onto the empty image at the specified coordinates
            empty_image[y1:y2, x1:x2] = colored_image_resized
            
            # Draw a yellow rectangle around the pasted image
            cv2.rectangle(empty_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
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
cluster_image_path = os.path.join(ID_path, "Clustered_image.jpg")
cv2.imwrite(cluster_image_path, empty_image)





########## ORIGINAL IMAGE  ##########

original_image = cv2.imread(i_path + "VTA.tif", cv2.IMREAD_GRAYSCALE)
original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
plt.imshow(original_image)



# Loop through the rows of the 'data' DataFrame
for _, row in data.iterrows():
    cell_number = row['Cell']
    cluster = row['Cluster']
    coordinates = eval(row['cell_positions'])
    x1, y1, x2, y2 = coordinates
    
    # Construct the image filename based on cell_number
    image_filename = f"{cell_number}.tif"
    
    # Check if the image file exists in the directory
    if image_filename in os.listdir(color_path):
        input_image = os.path.join(color_path, image_filename)
        
        # Check if the input image file exists
        if os.path.isfile(input_image):
            # Read the colored image
            colored_image = cv2.imread(input_image)
            
            # Calculate the dimensions of the region where the image will be pasted
            overlay_width = x2 - x1
            overlay_height = y2 - y1
            
            # Resize the colored image to fit the overlay region
            colored_image_resized = cv2.resize(colored_image, (overlay_width, overlay_height))
            
            # Overlay the resized colored image onto the empty image at the specified coordinates
            original_image[y1:y2, x1:x2] = colored_image_resized
            
            # Draw a yellow rectangle around the pasted image
            cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
            # Write the cell number at the top of the box in red
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_color = (0, 0, 255)
            font_thickness = 2
            text_size = cv2.getTextSize(str(cell_number), font, font_scale, font_thickness)[0]
            text_x = x1 + (overlay_width - text_size[0]) // 2
            text_y = y1 - 5
            cv2.putText(original_image, str(cell_number), (text_x, text_y), font, font_scale, font_color, font_thickness)

# Save the resulting image
cluster_image_path = os.path.join(ID_path, "Original_Clustered_image.jpg")
cv2.imwrite(cluster_image_path, original_image)

        
        
        
      
