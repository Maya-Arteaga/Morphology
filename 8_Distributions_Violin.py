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
from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn

# Load the data from the CSV file
i_path = "/Users/juanpablomayaarteaga/Desktop/Hilus/Prepro/"
o_path = set_path(i_path + "/Output_images/")
csv_path = set_path(o_path + "/Merged_Data/")
Distributions_path = set_path(o_path + "Plots/Distributions/")
Violin_path = set_path(o_path + "Plots/Violins/")

data = pd.read_csv(csv_path + "Morphology_PCA_UMAP_HDBSCAN_10.csv")



####DISTRIBUTION PLOTS


# List of variables to create distribution plots for
variables = ["Area_soma", "Perimeter_soma", "Circularity_soma", "soma_compactness", "soma_orientation",
             "soma_feret_diameter", "soma_eccentricity", "soma_aspect_ratio", "End_Points", "Junctions",
             "Branches", "Initial_Points", "Total_Branches_Length", "ratio_branches", "polygon_area",
             "polygon_perimeters", "polygon_compactness", "polygon_eccentricities", "polygon_feret_diameters",
             "polygon_orientations", "cell_area", "cell_perimeter", "cell_circularity", "cell_compactness",
             "cell_orientation", "cell_feret_diameter", "cell_eccentricity", "cell_aspect_ratio",
             "sholl_max_distance", "sholl_crossing_processes", "sholl_num_circles", "cell_solidity",
             "cell_convexity", "UMAP_1", "UMAP_2"]


"""
variables = ["UMAP_1", "UMAP_2"]

#DISTRIBUTIONS

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


"""

# VIOLIN PLOTS


# Column for categories
category_column = "categories"

# Define colors and labels for each category
colors = {'VEH_ESC': 'red', 'CNEURO1_ESC': 'yellow', 'VEH_SS': 'green', "CNEURO-01_ESC": "orange", "CNEURO1_SS": "blue"}
hue_order = ["VEH_SS", "VEH_ESC","CNEURO-01_ESC",  "CNEURO1_ESC", "CNEURO1_SS"]
labels = {"VEH_SS": "VEH SS", "CNEURO1_ESC": "CNEURO 1.0", "VEH_ESC": "ESC", "CNEURO-01_ESC": "CNEURO 0.1", "CNEURO1_SS": "CNEURO 1.0 SS"}


for variable in variables:
    modified_variable = variable.replace("_", " ").title()
    # Perform Kruskal-Wallis test
    kw_stat, kw_p_value = kruskal(*[data[data[category_column] == label][variable] for label in hue_order])
    
    # Perform Dunn post hoc test
    dunn_results = posthoc_dunn(data, val_col=variable, group_col=category_column)
    print(dunn_results)


# VIOLIN PLOTS

# Create violin plots for each variable with the three categories
for variable in variables:
    plt.figure(figsize=(12, 8))
    
    # Modified variable name without underscores and with capitalized words
    modified_variable = variable.replace("_", " ").title()
    
    # Create violin plot with dodge
    ax = sns.violinplot(x=category_column, y=variable, hue=category_column, data=data, palette=colors, order=hue_order, inner='quartile', alpha=0.2, dodge=False, legend=False)
    
    # Customize the color of percentile lines directly
    for line in ax.lines:
        line.set_color("white")
    
    # Add scatter dots with dodge
    sns.stripplot(x=category_column, y=variable, hue=category_column, data=data, palette=colors, order=hue_order, jitter=True, dodge=False, size=8, alpha=0.3, ax=ax, zorder=1, legend=False)
    
    # Remove spines and trim
    # sns.despine(ax=ax, left=True, trim=True)
    
    # Turn off both x-axis and y-axis grid lines
    ax.grid(False)
    
    # Customize x-axis labels to be in capital letters
    ax.set_xticklabels([label.replace("_", " ").title().upper() for label in hue_order])

    plt.style.use('dark_background')

    # Customize title
    plt.title(f'{modified_variable}', fontsize=17, fontweight='bold')
    
    plt.xlabel(category_column.upper(), fontsize=12, fontweight='bold')
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(Violin_path + f'{variable}_violin_plot.png', bbox_inches='tight', dpi=500)  
    plt.show()
    
    