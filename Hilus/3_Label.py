

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

Watershed_path= set_path(Preprocess_path+"/7_Watershed/")

Edited_path= set_path(Preprocess_path+"/9_Edited/")

Labeled_path= set_path(Preprocess_path+"/10_Labeled/")



#tissues = ["T1", "T2", "T3", "T7"]
tissues = ["T2"]


for t in tissues:
    img = f"R1_VEH_SS_{t}_HILUS_FOTO1.jpg"
    image_path = Edited_path + img
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    
    
    # Set parameters
    min_area = 2000
    max_area = 2400000
    #max_area=6000
    num_cells_filtered2 = 0
    
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
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
    cv2.imwrite(Labeled_path + img, color_image)


        
