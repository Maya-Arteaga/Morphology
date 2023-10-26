#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 16:59:08 2023

@author: juanpablomayaarteaga
"""




import cv2
import matplotlib.pyplot as plt
from morpho import gammaCorrection, set_path
from skimage import filters
import numpy as np
from skimage.segmentation import clear_border
from skimage import color





i_path="/Users/juanpablomayaarteaga/Desktop/Hilus/"

o_path= set_path(i_path+"/Output_images/")

Preprocess_path = set_path(o_path+"/Preprocess/")

Gamma_path= set_path(Preprocess_path+"/1_Gamma/")

Unsharp_path= set_path(Preprocess_path+"/2_Unsharp/")

Bright_path= set_path(Preprocess_path+"/3_Brigth/")

Gaussian_path= set_path(Preprocess_path+"/4_Gaussian/")

NLM_path = set_path(Preprocess_path+"/5_NLM/")

Otsu_path= set_path(Preprocess_path+"/6_Otsu/")

Watershed_path= set_path(Preprocess_path+"/7_Watershed/")

Prelabeled_path= set_path(Preprocess_path+"/8_Prelabeled/")










gamma = [3.0]
blurr = [2]
alpha = [10.0]
sigma = [1.1]
dist = [0.10]
area= 50000




tissues = ["T1", "T2", "T3", "T7"]
#tissues = ["T2"]

for t in tissues:
    # Create the image file name using the current tissue type
    img=f"R1_VEH_SS_{t}_HILUS_FOTO1.jpg"
    image_path = i_path + img
    
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    
    height, width, channels = image.shape
    print(f"Image Shape: Height={height}, Width={width}, Channels={channels}")
    
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    
    inverted_image = cv2.subtract(255, gray_image)
    #plt.imshow(inverted_image)
    
    
    
    
    for g in gamma:
        gammaImg = gammaCorrection(inverted_image, g)
        cv2.imwrite(Gamma_path + img, gammaImg)
    
        for b in blurr:
            blurred = cv2.GaussianBlur(gammaImg, (0, 0), b)
            unsharp_mask = cv2.addWeighted(gammaImg, 1.5, blurred, -0.5, 0)
            sharp_image = cv2.add(gammaImg, unsharp_mask)
            cv2.imwrite(Unsharp_path + img, sharp_image)
            
            # Apply non-local means denoising
            denoised_image = cv2.fastNlMeansDenoising(sharp_image, h=10, templateWindowSize=7, searchWindowSize=21)
            cv2.imwrite(NLM_path + img, denoised_image)
    
    
    
            for a in alpha:
                bright_img = cv2.convertScaleAbs(sharp_image, alpha=a, beta=0)
                cv2.imwrite(Bright_path + img, bright_img)
    
                for s in sigma:
                    gaussian_array = filters.gaussian(bright_img, sigma=s)
                    gaussian_image = (gaussian_array * 255).astype(np.uint8)
                    cv2.imwrite(Gaussian_path + img, gaussian_image)
    

    
                    ret, Otsu_image = cv2.threshold(gaussian_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    cv2.imwrite(Otsu_path + img, Otsu_image)
                        
                    for d in dist:
                        # Apply watershed
                        reference_image = cv2.cvtColor(gaussian_image, cv2.COLOR_GRAY2BGR)
                        ret1, thresh = cv2.threshold(Otsu_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        kernel = np.ones((3, 3), np.uint8)
                        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
                        opening = clear_border(opening)
                        
                        sure_bg = cv2.dilate(opening, kernel, iterations=1)
                        
                        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
                        
                        ret2, sure_fg = cv2.threshold(dist_transform, d * dist_transform.max(), 255, 0)
                        sure_fg = np.uint8(sure_fg)
                        unknown = cv2.subtract(sure_bg, sure_fg)
                        ret3, markers = cv2.connectedComponents(sure_fg)
                        markers = markers + 10
                        markers[unknown == 255] = 0
                        
                        plt.imshow(markers)
                        plt.imshow(reference_image)
                        
                        # Apply watershed
                        markers = cv2.watershed(reference_image, markers)
                        reference_image[markers == -1] = [0, 0, 120]
                        
                        
                        gray_markers= (markers * 255).astype(np.uint8)
 

                        
                        # Apply thresholding to the grayscale image
                        _, binary_image = cv2.threshold(gray_markers, 245, 255, cv2.THRESH_BINARY)
                        
                        # Invert the binary image
                        inverted_binary_image = 255 - binary_image
                        
                        #####Removing white frame
                        # Get image dimensions
                        height, width = inverted_binary_image.shape
                        # Define the frame width (number of pixels to convert to background)
                        frame_width = 1  # You can adjust this value
                        # Convert the top row to background
                        inverted_binary_image[0:frame_width, :] = 0
                        # Convert the bottom row to background
                        inverted_binary_image[height - frame_width:, :] = 0
                        # Convert the left column to background
                        inverted_binary_image[:, 0:frame_width] = 0
                        # Convert the right column to background
                        inverted_binary_image[:, width - frame_width:] = 0

                        # Save the filled image
                        cv2.imwrite(Watershed_path + img, inverted_binary_image)

                        
                        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted_binary_image, connectivity=4)
                        
                        
                        # Set parameters
                        min_area = 500
                        max_area = 2400000
                        #max_area=6000
                        num_cells_filtered2 = 0
                        
                        color_image = cv2.cvtColor(inverted_binary_image, cv2.COLOR_GRAY2BGR)
                        
                        labeled_cells = 0
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
                                cv2.rectangle(color_image, (bounding_box_x, bounding_box_y), (bounding_box_x + bounding_box_width, bounding_box_y + bounding_box_height), (0, 255, 255), 2)
                                labeled_cells += 1
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                bottom_left = (bounding_box_x, bounding_box_y + bounding_box_height + 20)
                                font_scale = 0.5
                                color = (0, 0, 255)
                                thickness = 2
                                cv2.putText(color_image, str(labeled_cells), bottom_left, font, font_scale, color, thickness)
                        
                        # Save the original image with red rectangles around objects
                        cv2.imwrite(Prelabeled_path + img, color_image)

                        
                        

                 


















