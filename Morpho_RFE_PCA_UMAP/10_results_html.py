#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 17:12:31 2023
@author: juanpablomayaarteaga
"""

import pandas as pd
from morpho import set_path
from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn
from statsmodels.stats import multitest
from IPython.display import display

# Load the data from the CSV file
i_path = "/Users/juanpablomayaarteaga/Desktop/Hilus/Prepro/"
o_path = set_path(i_path + "/Output_images/")
csv_path = set_path(o_path + "Merged_Data/")

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

# Loop through variables
for variable in variables:
    # Perform Kruskal-Wallis test
    kruskal_result = kruskal(*[data[data[category_column] == category][variable] for category in ["VEH_SS", "CNEURO1_ESC", "VEH_ESC"]])

    # Perform Dunn test for pairwise comparisons
    dunn_results = posthoc_dunn(data, val_col=variable, group_col=category_column, p_adjust="holm")

    # Extract p-values for the specified comparisons
    p_veh_ss_cneuro1_esc = dunn_results.loc["VEH_SS", "CNEURO1_ESC"]
    p_veh_ss_veh_esc = dunn_results.loc["VEH_SS", "VEH_ESC"]
    p_cneuro1_esc_veh_esc = dunn_results.loc["CNEURO1_ESC", "VEH_ESC"]

    # Add results to the dataframe
    result_df = result_df.append({
        "Variable": variable,
        "Kruskal-Wallis": kruskal_result.pvalue,
        "VEH_SS vs CNEURO1_ESC Dunn test": p_veh_ss_cneuro1_esc,
        "VEH_SS vs VEH_ESC Dunn test": p_veh_ss_veh_esc,
        "CNEURO1_ESC vs VEH_ESC Dunn test": p_cneuro1_esc_veh_esc
    }, ignore_index=True)

# Perform Bonferroni and FDR corrections for Kruskal-Wallis
p_values_kruskal = result_df["Kruskal-Wallis"]
bonferroni_correction_kruskal = multitest.multipletests(p_values_kruskal, method='bonferroni')[1]
fdr_correction_kruskal = multitest.multipletests(p_values_kruskal, method='fdr_bh')[1]

# Add corrected p-values to the dataframe for Kruskal-Wallis
result_df["Kruskal-Wallis Bonferroni"] = bonferroni_correction_kruskal
result_df["Kruskal-Wallis FDR"] = fdr_correction_kruskal

# Perform Bonferroni and FDR corrections for Dunn test
p_values_dunn = result_df[["VEH_SS vs CNEURO1_ESC Dunn test", "VEH_SS vs VEH_ESC Dunn test", "CNEURO1_ESC vs VEH_ESC Dunn test"]]
bonferroni_correction_dunn = multitest.multipletests(p_values_dunn.values.flatten(), method='bonferroni')[1]
fdr_correction_dunn = multitest.multipletests(p_values_dunn.values.flatten(), method='fdr_bh')[1]

# Reshape corrected p-values and add them to the dataframe for Dunn test
bonferroni_correction_dunn_reshaped = bonferroni_correction_dunn.reshape(p_values_dunn.shape)
fdr_correction_dunn_reshaped = fdr_correction_dunn.reshape(p_values_dunn.shape)

result_df["VEH_SS vs CNEURO1_ESC Dunn test Bonferroni"] = bonferroni_correction_dunn_reshaped[:, 0]
result_df["VEH_SS vs VEH_ESC Dunn test Bonferroni"] = bonferroni_correction_dunn_reshaped[:, 1]
result_df["CNEURO1_ESC vs VEH_ESC Dunn test Bonferroni"] = bonferroni_correction_dunn_reshaped[:, 2]

result_df["VEH_SS vs CNEURO1_ESC Dunn test FDR"] = fdr_correction_dunn_reshaped[:, 0]
result_df["VEH_SS vs VEH_ESC Dunn test FDR"] = fdr_correction_dunn_reshaped[:, 1]
result_df["CNEURO1_ESC vs VEH_ESC Dunn test FDR"] = fdr_correction_dunn_reshaped[:, 2]

# Highlight values less than 0.01 in red, and between 0.01 and 0.05 in orange
html_result_df = result_df.style.applymap(lambda x: 'color: red' if isinstance(x, (float, int)) and float(x) < 0.01 else ('color: purple' if isinstance(x, (float, int)) and 0.01 <= float(x) <= 0.05 else ''))

# Save the styled result to an HTML file
html_result_df.to_html('Statistical_Results.html')


# Save the styled result to an HTML file
html_result_df.to_html(csv_path +'Statistical_Results.html')


# Save the results to a CSV file
result_df.to_csv(csv_path + "Morphology_Statistical_Results.csv", index=False)





