#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 19:45:21 2023

@author: juanpablomayaarteaga


ESTE ARCHIVO SE DEBE DE CORRER EN LA TERMINAL

ALIAS "hilus"

comando: python nombre_del_py



"""

import cv2
import numpy as np
from morpho import set_path




i_path="/Users/juanpablomayaarteaga/Desktop/Hilus/"

o_path= set_path(i_path+"/Output_images/")

Preprocess_path = set_path(o_path+"/Preprocess/")

Watershed_path= set_path(Preprocess_path+"/7_Watershed/")

Edited_path= set_path(Preprocess_path+"/9_Edited/")


tissues = ["T1", "T2", "T3", "T7"]
#tissues = ["T2"]




# Define global variables
drawing = False
value = 0  # Initialize the value to switch between background and foreground

def draw_circle(event, x, y, flags, param):
    global drawing, value

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        if value == 0:
            cv2.circle(binary_image, (x, y), 4, 255, -1)  # Convert background to foreground (change 0 to 255)
        else:
            cv2.circle(binary_image, (x, y), 4, 0, -1)  # Convert foreground to background (change 255 to 0)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False


if __name__ == "__main__":
    
    
    
    for t in tissues:
        img = f"R1_VEH_SS_{t}_HILUS_FOTO1.jpg"
        image_path = Watershed_path + img
        binary_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        
        if binary_image is None:
            print("Error: Unable to load the binary image.")
            exit()
    
        cv2.namedWindow("Interactive Binary Image Editor")
        cv2.setMouseCallback("Interactive Binary Image Editor", draw_circle)
    
        while True:
            cv2.imshow("Interactive Binary Image Editor", binary_image)
            key = cv2.waitKey(1) & 0xFF
    
            if key == ord('f'):
                value = 0  # Set the mode to convert background to foreground
            elif key == ord('b'):
                value = 1  # Set the mode to convert foreground to background
            elif key == 27:
                break  # Exit the program when the Esc key is pressed
    
        # Save the edited image
        cv2.imwrite(Edited_path + img, binary_image)
    
        print("Edited image saved at:", Edited_path + img)
    
    cv2.destroyAllWindows()
    print("Edited image saved at:", Edited_path + img)

    
    
