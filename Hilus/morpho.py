#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 18:02:46 2023

@author: juanpablomayaarteaga
"""

import os
import tifffile as tiff
import re

def name_2_num(path=None):
    if path is None:
        raise ValueError("The 'path' argument is required.")
    
        
    for filename in os.listdir(path):
        if filename.endswith(".tif"):
            original_path = os.path.join(path, filename)
            just_number = re.findall(r"\d+", filename)[0] + ".tif"  
            new_path = os.path.join(path, just_number)
            os.rename(original_path, new_path) 
            
    
        
        
        
        
        
def save_tif(filename, name=None, path=None, variable=None):
    if name is None:
        name = "add_name"  
    if path is None:
        path = "default/path"
    if variable is None:
        raise ValueError("The 'variable' argument is required.")

    output_filename = filename[:-4] + name
    output_path = os.path.join(path, output_filename)
    tiff.imwrite(output_path, variable)
    


def set_path(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        
    return path


import numpy as np
import cv2



def gammaCorrection(image, gamma):

    table = [((i / 255) ** gamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(image, table)



from PIL import Image

def padding(image_path, padding_rows, padding_cols):
    # Load the image
    image = Image.open(image_path).convert("L")

    # Get the dimensions of the original image
    width, height = image.size

    # Create a new image with the desired dimensions
    new_width = width + 2 * padding_cols
    new_height = height + 2 * padding_rows
    new_image = Image.new("L", (new_width, new_height), 0)

    # Paste the original image into the new image, leaving the padding area empty
    new_image.paste(image, (padding_cols, padding_rows))

    # Save the modified image
    new_image.save("padded_image.tiff")
    padded_image=np.array(new_image)

    print("Padding added successfully.")
    return padded_image

def find_contours(image):
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def polygon(image):
    # Find the contours in the binary image
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Approximate the contours to reduce points
    epsilon = 0.02 * cv2.arcLength(contours[0], True)
    approx = cv2.approxPolyDP(contours[0], epsilon, True)

    # Create a convex hull around the approximated contour
    hull = cv2.convexHull(approx)

    # Create a blank image
    polygon_image = np.zeros_like(image, dtype=np.uint8)

    # Draw the convex hull on the blank image
    cv2.fillPoly(polygon_image, [hull], (255))

    return polygon_image

def calculate_area(image):
    # Read the  image
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    # Apply thresholding to convert to binary image
    _, thresholded_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)

    # Count the white pixels (foreground)
    area = cv2.countNonZero(thresholded_image)

    return area


from scipy.ndimage import label, generate_binary_structure

def erase(image, elements):
    
    x=elements
    structure=generate_binary_structure(2,2)
    labeled_image, num_labels = label(image, structure)
    label_sizes=np.bincount(labeled_image.ravel())
    mask=label_sizes[labeled_image]>x
    erased_image=np.where(mask, image, 0)
    
    erased_image = (erased_image * 255).astype(np.uint8)
    
    return erased_image





def detect_and_color(image):
    # Padding
    img_padded = np.pad(image, 1)

    # Detect columns and rows
    n_row = np.shape(img_padded)[0]
    n_col = np.shape(img_padded)[1]

    # Empty matrix
    M = np.zeros((n_row - 2, n_col - 2), dtype=np.int8)

    # Create an empty colored image
    colored_image = np.zeros((M.shape[0], M.shape[1], 3), dtype=np.uint8)

    # Detect neighbors and assign colors
    for i in range(1, n_row):
        for j in range(1, n_col):
            if img_padded[i, j] != 0:
                M[i-1, j-1] = np.count_nonzero(img_padded[i-1:i+2, j-1:j+2]) - 1  # matriz 3x3

                value = M[i-1, j-1]
                if value == 1:
                    colored_image[i-1, j-1] = (0, 0, 255)  # Blue - Red (tif)
                elif value == 2:
                    colored_image[i-1, j-1] = (255, 0, 0)  # Red - Blue (tif)
                elif value >= 3:
                    colored_image[i-1, j-1] = (0, 255, 0)  # Green - Green (tif)
            

    # Converting to RGB
    colored_image = cv2.cvtColor(colored_image, cv2.COLOR_BGR2RGB)

    return M, colored_image





def count_branches(image):
    image= image/255
    height = len(image)
    width = len(image[0])
    visited = [[False] * width for _ in range(height)]
    label = 0
    object_count = 0

    def dfs(row, col):
        visited[row][col] = True

        # Check neighbors (horizontally, vertically, and diagonally)
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                new_row = row + dr
                new_col = col + dc

                # Skip out-of-bounds pixels
                if new_row < 0 or new_row >= height or new_col < 0 or new_col >= width:
                    continue

                # If connected foreground pixel is found, perform DFS on it
                if image[new_row][new_col] == 1 and not visited[new_row][new_col]:
                    dfs(new_row, new_col)

    for row in range(height):
        for col in range(width):
            if image[row][col] == 1 and not visited[row][col]:
                label += 1
                object_count += 1
                dfs(row, col)

    return object_count


from scipy.ndimage import label, generate_binary_structure

def count(image):
    
    labels, num_objects = label(image)
    
    return num_objects


def detect_features(image, soma_image):
    
    kernel = np.ones((3,3),np.uint8)
    soma_image = cv2.dilate(soma_image, kernel, iterations=1)
    # Padding
    img_padded = np.pad(image, 1)

    # Detect columns and rows
    n_row = np.shape(img_padded)[0]
    n_col = np.shape(img_padded)[1]

    # Empty matrix
    M = np.zeros((n_row - 2, n_col - 2), dtype=np.int8)

    # Create an empty colored image
    colored_image = np.zeros((M.shape[0], M.shape[1], 3), dtype=np.uint8)

    # Detect neighbors and assign colors
    for i in range(1, n_row):
        for j in range(1, n_col):
            if img_padded[i, j] != 0:
                M[i-1, j-1] = np.count_nonzero(img_padded[i-1:i+2, j-1:j+2]) - 1  # matriz 3x3

                value = M[i-1, j-1]
                if value == 1:
                    colored_image[i-1, j-1] = (0, 0, 255)  # Blue - Red (tif)
                elif value == 2:
                    colored_image[i-1, j-1] = (255, 0, 0)  # Red - Blue (tif)
                elif value >= 3:
                    colored_image[i-1, j-1] = (0, 255, 0)  # Green - Green (tif)

    # Converting to RGB
    colored_image = cv2.cvtColor(colored_image, cv2.COLOR_BGR2RGB)

    # Identify initial points in contact with soma
    initial_points = cv2.bitwise_and(image, soma_image)
    colored_image[initial_points > 0] = (200, 200, 0)  # Yellow (tif)

    return M, colored_image

