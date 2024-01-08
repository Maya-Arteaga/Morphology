#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 00:14:22 2023

@author: juanpablomayaarteaga
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 02:42:55 2023

@author: juanpablomayaarteaga
"""

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



n_neighbors=10


i_path = "/Users/juanpablomayaarteaga/Desktop/Hilus/Prepro/"
o_path = os.path.join(i_path, "Output_images/")
csv_path = os.path.join(o_path, "Merged_Data")
csv_file = f"Morphology_PCA_UMAP_HDBSCAN_{n_neighbors}.csv"
data = os.path.join(csv_path, csv_file)

# Read the CSV file
df = pd.read_csv(data)


subject = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10"]
group = ["CNEURO1", "VEH", "CNEURO-01"]
treatment = ["ESC", "SS"]
tissue = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10"]

# Iterate over categories and create separate CSV files if the group exists
for s in subject:
    for g in group:
        for tr in treatment:
            for ti in tissue:
                # Check if the combination exists in the DataFrame
                combination_exists = not df[
                    (df['subject'] == s) &
                    (df['group'] == g) &
                    (df['treatment'] == tr) &
                    (df['tissue'] == ti)
                ].empty
                
                if combination_exists:
                    # Filter the dataframe based on the given criteria
                    filtered_data = df[
                        (df['subject'] == s) &
                        (df['group'] == g) &
                        (df['treatment'] == tr) &
                        (df['tissue'] == ti)
                    ]
                    # Save the filtered data to a separate CSV file
                    output_file = f"{s}_{g}_{tr}_{ti}_Morphology_PCA_UMAP_HDBSCAN_{n_neighbors}.csv"
                    output_path = os.path.join(csv_path, output_file)
                    filtered_data.to_csv(output_path, index=False)






import os
import shutil

# Iterate over the CSV files and move each to its directory
for s in subject:
    for g in group:
        for tr in treatment:
            for ti in tissue:
                csv_file_name = f"{s}_{g}_{tr}_{ti}_Morphology_PCA_UMAP_HDBSCAN_{n_neighbors}.csv"
                source_file = os.path.join(csv_path, csv_file_name)
                target_directory = os.path.join(o_path, f"{s}_{g}_{tr}_{ti}_CA1_FOTO1_PROCESSED", "Data")
                
                if os.path.exists(source_file) and os.path.exists(target_directory):
                    shutil.copy2(source_file, target_directory)
                    os.remove(source_file)  # Remove the original file

                else:
                    continue
                    #print(f"File '{csv_file_name}' or directory '{target_directory}' does not exist.")



