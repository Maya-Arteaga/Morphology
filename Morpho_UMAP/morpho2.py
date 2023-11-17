
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 18:02:46 2023

@author: juanpablomayaarteaga
"""

import cv2
import numpy as np
from skimage.measure import label, regionprops

def sholl_circles(image):
    # Extract the red channel
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Define the lower and upper bounds for the red color
    lower_red = np.array([0, 0, 200], dtype=np.uint8)  # Adjust the threshold as needed
    upper_red = np.array([100, 100, 255], dtype=np.uint8)  # Adjust the threshold as needed

    # Create a mask to filter out the white values
    mask_white = cv2.inRange(img, (200, 200, 200), (255, 255, 255))

    # Create a mask to filter out the red values
    mask_red = cv2.inRange(img, lower_red, upper_red)

    # Combine the masks
    red_pixels = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask_white))

    # Connected components labeling
    centroid, num_objects = label(red_pixels, connectivity=2, background=0, return_num=True)

    # Get region properties
    props = regionprops(centroid)

    # Calculate the distance between the centroid and the four vertices of the image
    max_distance = max(
        np.linalg.norm([props[0].centroid[1], props[0].centroid[0]]),  # Top-left vertex
        np.linalg.norm([props[0].centroid[1], img.shape[0] - props[0].centroid[0]]),  # Bottom-left vertex
        np.linalg.norm([img.shape[1] - props[0].centroid[1], props[0].centroid[0]]),  # Top-right vertex
        np.linalg.norm([img.shape[1] - props[0].centroid[1], img.shape[0] - props[0].centroid[0]])  # Bottom-right vertex
    )

    # Pad the image to be exactly the size of the circumscribed circle
    padding = int(max_distance)  # Round down to the nearest integer
    img_padded = np.pad(img, ((padding, padding), (padding, padding), (0, 0)))

    # Create an empty colored image
    circle_image = np.zeros_like(img_padded)

    # Draw a blue point at the centroid
    blue_point_coords = (
        int(props[0].centroid[1] + padding), 
        int(props[0].centroid[0] + padding)
    )  # Row and column coordinates
    circle_image[blue_point_coords[1], blue_point_coords[0]] = [255, 0, 0]

    # Draw red growing circles with the centroid as the center
    for radius in range(10, int(max_distance), 10):
        cv2.circle(circle_image, blue_point_coords, radius, (0, 0, 255), 2)

    # Overlap the padded input image and the colored image bitwise
    result_image = cv2.bitwise_or(img_padded, circle_image)

    # Detect if the colored image and the padded input image touch at some points
    touching_points = cv2.bitwise_and(img_padded, circle_image)
    
    # Count the number of touching points
    num_touching_points = np.max(label(touching_points))
    
    # Convert the touching points to green
    result_image[np.where((touching_points == [0, 0, 255]).all(axis=2))] = [0, 255, 0]  # Green color
    sholl_image = result_image
    #sholl_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    #circle_image = cv2.cvtColor(circle_image, cv2.COLOR_RGB2BGR)
    
    return sholl_image, max_distance, num_touching_points, circle_image

#colored_image
