import os
import pandas as pd
from morpho import set_path
from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn
from statsmodels.stats import multitest
from IPython.display import display

# Function to apply color-coding based on the specified conditions
def apply_color(x):
    if isinstance(x, (float, int)) and x <= 0.001:
        return 'color: red'
    elif isinstance(x, (float, int)) and 0.001 < x <= 0.01:
        return 'color: purple'
    elif isinstance(x, (float, int)) and 0.01 < x <= 0.05:
        return 'color: blue'
    else:
        return ''

# Load the data from the CSV file
i_path = "/Users/juanpablomayaarteaga/Desktop/Hilus/Prepro/"
o_path = set_path(i_path + "/Output_images/")
csv_path = set_path(o_path + "Merged_Data/")

data = pd.read_csv(csv_path + "Morphology_PCA_UMAP_HDBSCAN_10.csv")

# Categories to compare: VEH_SS, VEH_ESC, CNEURO-01_ESC, CNEURO1_ESC, CNEURO1_SS
categories = ["VEH_SS", "VEH_ESC", "CNEURO-01_ESC", "CNEURO1_ESC", "CNEURO1_SS"]

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
result_df = pd.DataFrame(columns=["Variable", "Kruskal-Wallis"] + [f"{cat1} vs {cat2} Dunn test" for i, cat1 in enumerate(categories) for j, cat2 in enumerate(categories) if i < j])

# Loop through variables
for variable in variables:
    # Perform Kruskal-Wallis test
    kruskal_result = kruskal(*[data[data["categories"] == category][variable] for category in categories])

    # Perform Dunn test for pairwise comparisons
    dunn_results = posthoc_dunn(data, val_col=variable, group_col="categories", p_adjust="holm")

    # Extract p-values for the specified comparisons
    p_values = {}
    for i, cat1 in enumerate(categories):
        for j, cat2 in enumerate(categories):
            if i < j:
                p_values[f"{cat1} vs {cat2}"] = dunn_results.loc[cat1, cat2]

    # Add results to the dataframe
    result_df = result_df.append({
        "Variable": variable,
        "Kruskal-Wallis": kruskal_result.pvalue,
        **{f"{pair} Dunn test": p_values[pair] for pair in p_values}
    }, ignore_index=True)

    # Perform Bonferroni and FDR corrections for Dunn test
    for i, cat1 in enumerate(categories):
        for j, cat2 in enumerate(categories):
            if i < j:
                p_values_dunn = result_df[f"{cat1} vs {cat2} Dunn test"]
                bonferroni_correction_dunn = multitest.multipletests(p_values_dunn, method='bonferroni')[1]
                fdr_correction_dunn = multitest.multipletests(p_values_dunn, method='fdr_bh')[1]

                result_df[f"{cat1} vs {cat2} Dunn test Bonferroni"] = bonferroni_correction_dunn
                result_df[f"{cat1} vs {cat2} Dunn test FDR"] = fdr_correction_dunn

# Save the styled result to an HTML file
html_result_df = result_df.style.applymap(apply_color)
html_result_df.to_html(csv_path + 'Statistical_Results.html')

# Save the results to a CSV file
result_df.to_csv(csv_path + "Morphology_Statistical_Results.csv", index=False)
