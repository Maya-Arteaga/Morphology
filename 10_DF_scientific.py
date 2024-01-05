#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 16:51:52 2023

@author: juanpablomayaarteaga
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from morpho import set_path
from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn
from statsmodels.stats import multitest
from IPython.display import display

# Load the data from the CSV file
i_path = "/Users/juanpablomayaarteaga/Desktop/Hilus/Prepro/"
o_path = set_path(i_path + "/Output_images/")
csv_path = set_path(o_path + "/Merged_Data/")
Distributions_path = set_path(o_path + "Plots/Distributions/")
Violin_path = set_path(o_path + "Plots/Violins/")

data = pd.read_csv(csv_path + "Morphology_PCA_UMAP_HDBSCAN_15.csv")

# Categories to compare: VEH_SS, CNEURO1_ESC, VEH_ESC
category_column = "categories"

# Variables
variables = ["Area_soma", "Perimeter_soma", "Circularity_soma", "soma_compactness", "soma_orientation",
             "soma_feret_diameter", "soma_eccentricity", "soma_aspect_ratio", "End_Points", "Junctions",
             "Branches", "Initial_Points", "Total_Branches_Length", "ratio_branches", "polygon_area",
             "polygon_perimeters", "polygon_compactness", "polygon_eccentricities", "polygon_feret_diameters",
             "polygon_orientations", "cell_area", "cell_perimeter", "cell_circularity", "cell_compactness",
             "cell_orientation", "cell_feret_diameter", "cell_eccentricity", "cell_aspect_ratio",
             "sholl_max_distance", "sholl_crossing_processes", "sholl_num_circles", "cell_solidity",
             "cell_convexity", "UMAP_1", "UMAP_2"]

# Create an empty dataframe
result_df = pd.DataFrame(columns=["Variable", "Kruskal-Wallis", "VEH_SS vs CNEURO1_ESC Dunn test",
                                   "VEH_SS vs VEH_ESC Dunn test", "CNEURO1_ESC vs VEH_ESC Dunn test"])

for variable in variables:
    # Perform Kruskal-Wallis test
    kruskal_result = kruskal(*[data[data[category_column] == category][variable] for category in ["VEH_SS", "CNEURO1_ESC", "VEH_ESC"]])

    # Perform Dunn test for pairwise comparisons
    dunn_results = posthoc_dunn(data, val_col=variable, group_col=category_column, p_adjust="holm")

    # Extract p-values for the specified comparisons
    p_veh_ss_cneuro1_esc = dunn_results.loc["VEH_SS", "CNEURO1_ESC"]
    p_veh_ss_veh_esc = dunn_results.loc["VEH_SS", "VEH_ESC"]
    p_cneuro1_esc_veh_esc = dunn_results.loc["CNEURO1_ESC", "VEH_ESC"]
    
    # Format p-values in scientific notation with two digits
    p_veh_ss_cneuro1_esc_str = "{:.2e}".format(p_veh_ss_cneuro1_esc)
    p_veh_ss_veh_esc_str = "{:.2e}".format(p_veh_ss_veh_esc)
    p_cneuro1_esc_veh_esc_str = "{:.2e}".format(p_cneuro1_esc_veh_esc)
    
    # Add results to the dataframe
    result_df = result_df.append({
        "Variable": variable,
        "Kruskal-Wallis": "{:.2e}".format(kruskal_result.pvalue),
        "VEH_SS vs CNEURO1_ESC Dunn test": p_veh_ss_cneuro1_esc_str,
        "VEH_SS vs VEH_ESC Dunn test": p_veh_ss_veh_esc_str,
        "CNEURO1_ESC vs VEH_ESC Dunn test": p_cneuro1_esc_veh_esc_str
    }, ignore_index=True)


"""
# Highlight values less than 0.01 in red
styled_result_df = result_df.style.applymap(lambda x: 'color: red' if isinstance(x, (float, int)) and float(x) < 0.01 else '')


styled_result_df.to_html('styled_result.html')
"""


import os
print(os.getcwd())

# Display the result dataframe
#print(result_df)


# Save the results to a CSV file
result_df.to_csv(csv_path + "Morphology_Statistical_Results.csv", index=False)




