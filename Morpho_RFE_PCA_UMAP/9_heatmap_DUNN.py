import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from morpho import set_path
from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn
from matplotlib.colors import LinearSegmentedColormap

# Load the data from the CSV file
i_path = "/Users/juanpablomayaarteaga/Desktop/Hilus/Prepro/"
o_path = set_path(i_path + "/Output_images/")
csv_path = set_path(o_path + "/Merged_Data/")
Dunn_path = set_path(o_path + "Plots/Dunn/")

data = pd.read_csv(csv_path + "Morphology_PCA_UMAP_HDBSCAN_15.csv")

# Column for categories
category_column = "categories"
variables = ["UMAP_1", "UMAP_2"]


"""
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

# Define colors and labels for each category
colors = {'VEH_ESC': 'red', 'CNEURO1_ESC': 'blue', 'VEH_SS': 'green'}

# Specify the order of categories
hue_order = ['VEH_SS', 'CNEURO1_ESC', 'VEH_ESC']

plt.style.use('dark_background')



# Create heatmap for each variable with the three categories
for variable in variables:
    modified_variable = variable.replace("_", " ").title()
    # Perform Kruskal-Wallis test
    kw_stat, kw_p_value = kruskal(*[data[data[category_column] == label][variable] for label in hue_order])
    
    # Perform Dunn post hoc test
    dunn_results = posthoc_dunn(data, val_col=variable, group_col=category_column)
    
    # Plot the Dunn post hoc results with all digits after the decimal point
    plt.figure(figsize=(12, 8))
    ax_heatmap = sns.heatmap(dunn_results, annot=True, fmt=".2e", cmap="coolwarm_r", cbar=False,alpha=0.8, annot_kws={"size": 16})
    
    plt.style.use('dark_background')

    # Customize title with Kruskal-Wallis results
    title = f'{modified_variable} - Kruskal-Wallis: Stat={kw_stat:.2f}, p={kw_p_value:.2e}\nDunn Results:'
    ax_heatmap.set_title(title, fontsize=17, fontweight='bold')

    ax_heatmap.set_xlabel(category_column.upper(), fontsize=12, fontweight='bold')
    ax_heatmap.set_ylabel(category_column.upper(), fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    # Customize x-axis and y-axis labels to be in capital letters and without underscores
    ax_heatmap.set_xticklabels([label.replace("_", " ") for label in hue_order], fontsize=14, fontweight='bold')
    ax_heatmap.set_yticklabels([label.replace("_", " ") for label in hue_order], fontsize=14, fontweight='bold')

    # Save the heatmap
    plt.savefig(Dunn_path + f'{variable}_KW_DUNN.png', bbox_inches='tight', dpi=800)
    
    plt.show()
