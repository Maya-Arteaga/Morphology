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
Ridge_path = set_path(o_path + "Plots/Ridges/") 

data = pd.read_csv(csv_path + "Morphology_PCA_UMAP_HDBSCAN_15.csv")

# List of variables to create ridge plots for
variables = ["Area_soma", "Perimeter_soma", "Circularity_soma", "soma_compactness", "soma_orientation",
             "soma_feret_diameter", "soma_eccentricity", "soma_aspect_ratio", "End_Points", "Junctions",
             "Branches", "Initial_Points", "Total_Branches_Length", "ratio_branches", "polygon_area",
             "polygon_perimeters", "polygon_compactness", "polygon_eccentricities", "polygon_feret_diameters",
             "polygon_orientations", "cell_area", "cell_perimeter", "cell_circularity", "cell_compactness",
             "cell_orientation", "cell_feret_diameter", "cell_eccentricity", "cell_aspect_ratio",
             "sholl_max_distance", "sholl_crossing_processes", "sholl_num_circles", "cell_solidity",
             "cell_convexity", "UMAP_1", "UMAP_2"]

# Column for categories
category_column = "categories"

# Define colors and labels for each category
colors = {'VEH_ESC': 'red', 'CNEURO1_ESC': 'yellow', 'VEH_SS': 'green', "CNEURO-01_ESC": "orange", "CNEURO1_SS": "blue"}
hue_order = ["VEH_SS", "VEH_ESC", "CNEURO-01_ESC", "CNEURO1_ESC", "CNEURO1_SS"]
labels = {"VEH_SS": "VEH SS", "CNEURO1_ESC": "CNEURO 1.0", "VEH_ESC": "ESC", "CNEURO-01_ESC": "CNEURO 0.1",
          "CNEURO1_SS": "CNEURO 1.0 SS"}





import seaborn as sns
import matplotlib.pyplot as plt

# We generate a pd.Series with the mean UMAP_1 for each category
category_mean_serie = data.groupby('categories')['UMAP_1'].mean()
data['mean_UMAP_1'] = data['categories'].map(category_mean_serie)

# Define the specific colors for each category
colors = {'VEH_ESC': 'yellow', 'CNEURO1_ESC': 'red', 'VEH_SS': 'skyblue', "CNEURO-01_ESC": "orange", "CNEURO1_SS": "green"}
category_colors = [colors[category] for category in data['categories'].unique()]



# In the sns.FacetGrid class, the 'hue' argument is the one that will be represented by colors with 'palette'
g = sns.FacetGrid(data, row='categories', hue='mean_UMAP_1', aspect=8, height=0.95, palette=category_colors)

# Then we add the densities kdeplots for each category
g.map(sns.kdeplot, 'UMAP_1',
      bw_adjust=1, clip_on=False,
      fill=True, alpha=0.7, linewidth=1.0)

# Here we add a white line that represents the contour of each kdeplot
g.map(sns.kdeplot, 'UMAP_1', 
      bw_adjust=1, clip_on=False, 
      color="w", lw=2)

# Here we add a horizontal line for each plot
g.map(plt.axhline, y=0,
      lw=2, clip_on=False)

# We loop over the FacetGrid figure axes (g.axes.flat) and add the category as text with the right color
# Notice how ax.lines[-1].get_color() enables you to access the last line's color in each matplotlib.Axes
for i, ax in enumerate(g.axes.flat):
    ax.text(-15, 0.02, labels[hue_order[i]],
            fontweight='bold', fontsize=10,  # Font size of categories
            color=ax.lines[-1].get_color())

    # Set tick labels' font properties
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=15, fontweight='bold')

    # Remove the y-axis label
    ax.set_ylabel("")

# Adjust the vertical spacing between plots
g.fig.subplots_adjust(hspace=-0.05)

# Eventually, we remove axes titles, yticks, and spines
g.set_titles("")
g.set(yticks=[])
g.set(xticks=[])
g.despine(bottom=True, left=True)

# Set the overall title for the plot
g.fig.suptitle('',
               ha='right',
               fontsize=20,
               fontweight=20)

# Show the plot
plt.tight_layout()
plt.savefig(Ridge_path + 'ridge_plot.png', bbox_inches='tight', dpi=800) 
plt.show()






