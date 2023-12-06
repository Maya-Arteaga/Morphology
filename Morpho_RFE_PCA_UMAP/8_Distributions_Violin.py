#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 00:44:11 2023

@author: juanpablomayaarteaga
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from morpho import set_path

# Load the data from the CSV file
i_path = "/Users/juanpablomayaarteaga/Desktop/Hilus/Prepro/"
o_path = set_path(i_path + "/Output_images/")
csv_path = set_path(o_path + "/Merged_Data/")
Distributions_path = set_path(o_path + "Plots/Distributions/")
Violin_path = set_path(o_path + "Plots/Violins/")

data = pd.read_csv(csv_path + "Morphology.csv")




# List of variables to create distribution plots for
variables = ["Area_soma", "Perimeter_soma", "Circularity_soma", "soma_compactness", "soma_orientation",
             "soma_feret_diameter", "soma_eccentricity", "soma_aspect_ratio", "End_Points", "Junctions",
             "Branches", "Initial_Points", "Total_Branches_Length", "ratio_branches", "polygon_area",
             "polygon_perimeters", "polygon_compactness", "polygon_eccentricities", "polygon_feret_diameters",
             "polygon_orientations", "cell_area", "cell_perimeter", "cell_circularity", "cell_compactness",
             "cell_orientation", "cell_feret_diameter", "cell_eccentricity", "cell_aspect_ratio",
             "sholl_max_distance", "sholl_crossing_processes", "sholl_num_circles", "cell_solidity",
             "cell_convexity"]

#variables = ["Cluster_Labels"]
# Column for categories
category_column = "categories"

# Define colors and labels for each category
colors = {'VEH_ESC': 'green', 'CNEURO1_ESC': 'blue', 'VEH_SS': 'red'}
labels = {'VEH_ESC': 'Vehicle ESC', 'CNEURO1_ESC': 'CNEURO1 ESC', 'VEH_SS': 'Vehicle SS'}

# Create distribution plots for each variable with the three categories
for variable in variables:
    plt.figure(figsize=(10, 6))
    sns.histplot(data, x=variable, hue=category_column, kde=True, multiple="stack",
                 palette=colors, edgecolor=".3")
    plt.title(f'Distribution of {variable} by Categories')
    plt.xlabel(variable)
    plt.ylabel('Frequency')
    plt.legend(title=category_column, labels=[labels[key] for key in labels])
    plt.savefig(Distributions_path + f'{variable}_distribution.png')
    plt.show()



# Column for categories
category_column = "categories"

# Define colors and labels for each category
colors = {'VEH_ESC': 'red', 'CNEURO1_ESC': 'blue', 'VEH_SS': 'green'}
labels = {'VEH_ESC': 'Vehicle ESC', 'CNEURO1_ESC': 'CNEURO1 ESC', 'VEH_SS': 'Vehicle SS'}

# Specify the order of categories
hue_order = ['VEH_SS', 'CNEURO1_ESC', 'VEH_ESC']

# Create violin plots for each variable with the three categories
for variable in variables:
    plt.figure(figsize=(12, 8))
    ax = sns.violinplot(x=category_column, y=variable, data=data, palette=colors, order=hue_order)
    """
    # Add mean, median, standard deviation, and variance on top of the violins
    for category in hue_order:
        subset_data = data[data[category_column] == category]
        mean_val = subset_data[variable].mean()
        #median_val = subset_data[variable].median()
        #std_val = subset_data[variable].std()
        #var_val = subset_data[variable].var()

        ax.text(hue_order.index(category), mean_val + 0.05, f'Mean: {mean_val:.2f}', ha='center', va='center', color='white')
        #ax.text(hue_order.index(category), median_val - 0.1, f'Median: {median_val:.2f}', ha='center', va='center', color='white')
        #ax.text(hue_order.index(category), std_val + 0.1, f'Std Dev: {std_val:.2f}', ha='center', va='center', color='white')
        #ax.text(hue_order.index(category), var_val + 0.05, f'Variance: {var_val:.2f}', ha='center', va='top', color='white')
    """
    
    plt.title(f'{variable}')
    plt.xlabel(category_column)
    plt.ylabel(variable)
    plt.tight_layout()
    plt.savefig(Violin_path + f'{variable}_violin_plot.png')
    plt.show()
    
    
    
    