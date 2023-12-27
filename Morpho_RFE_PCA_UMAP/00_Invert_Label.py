#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 17:18:44 2023

@author: juanpablomayaarteaga
"""

import cv2
import os
from tkinter import filedialog
from morpho import set_path



# Define global variables
drawing = False
value = 0  # Initialize the value to switch between background and foreground
brush_size = 1  # Initialize the brush size




root_path = filedialog.askdirectory()
o_path = set_path(os.path.join(root_path, "Edited_images"))

for filename in os.listdir(root_path):
    base, extension = os.path.splitext(filename)

    # Skip files with names ending in "_labeled"
    if base.endswith("_labeled"):
        continue

    # Process only files with the specified extension
    if extension.lower() not in {".jpg", ".png", ".tif"}:
        continue

    # Read the file
    input_file = os.path.join(root_path, filename)
    image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    image = cv2.bitwise_not(image)


    # LABEL OBJECTS
    # Draw rectangles and save labeled images
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    min_area = 1000
    max_area = 2400000
    labeled_cells = 0
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            bounding_box_x = stats[i, cv2.CC_STAT_LEFT]
            bounding_box_y = stats[i, cv2.CC_STAT_TOP]
            bounding_box_width = stats[i, cv2.CC_STAT_WIDTH]
            bounding_box_height = stats[i, cv2.CC_STAT_HEIGHT]

            # Draw a rectangle around the object in the original image
            cv2.rectangle(color_image, (bounding_box_x, bounding_box_y),
                          (bounding_box_x + bounding_box_width, bounding_box_y + bounding_box_height),
                          (0, 255, 255), 2)
            labeled_cells += 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottom_left = (bounding_box_x, bounding_box_y + bounding_box_height + 20)
            font_scale = 0.5
            color = (0, 0, 255)
            thickness = 2
            cv2.putText(color_image, str(labeled_cells), bottom_left, font, font_scale, color, thickness)

    # Save the labeled image
    labeled_output_path = os.path.join(o_path, f"{base}_labeled.tif")
    cv2.imwrite(labeled_output_path, color_image)


    print("Labeled image saved at:", labeled_output_path)
