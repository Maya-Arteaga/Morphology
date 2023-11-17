#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 01:32:48 2023

@author: juanpablomayaarteaga
"""


from morpho import set_path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the data from the CSV file
i_path = "/Users/juanpablomayaarteaga/Desktop/Hilus/Prepro/"
o_path = set_path(i_path + "/Output_images/")
csv_path = set_path(o_path + "/Merged_Data/")
Plot_path = set_path(o_path + "Plots/")

data = pd.read_csv(csv_path + "Morphology.csv")

# Convert the "categories" column to categorical data type
data["categories"] = pd.Categorical(data["categories"])

# Selecting features for UMAP
features_to_assess= ["sholl_crossing_processes", "Junctions", "polygon_eccentricities",
                     "cell_compactness", "soma_aspect_ratio", "cell_aspect_ratio",
                     "cell_aspect_ratio", "cell_solidity",
                     "cell_perimeter", "soma_eccentricity", "polygon_perimeters",
                     "cell_area", "Initial_Points", "ratio_branches", "Total_Branches_Length",
                     "cell_circularity", "Branches", "Circularity_soma", "Area_soma", "cell_eccentricity"
                     ]

# Selecting the subset of features from the data
selected_data = data[features_to_assess]

# Create a pairplot
sns.pairplot(selected_data, hue="categories", corner=True)
plt.suptitle('Pairwise Relationships', y=1.02)
plt.savefig(Plot_path + "Pairwise Relationships.png", dpi=600)
plt.show()






# Selecting the subset of features from the data
selected_data = data[features_to_assess]

# Extract numerical columns for scaling
numerical_cols = [col for col in selected_data.columns if col != 'categories']


# Create a pairplot after scaling the data
sns.pairplot(selected_data, hue="categories", corner=True)
plt.suptitle('Pairwise Relationships (Scaled Data)', y=1.02)
plt.savefig(Plot_path + "Pairwise_Relationships_Scaled.png", dpi=600)
plt.show()




