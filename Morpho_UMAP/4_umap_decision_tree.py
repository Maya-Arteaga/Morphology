#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 17:19:37 2023

@author: juanpablomayaarteaga
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 22:41:44 2023

@author: juanpablomayaarteaga
"""

from morpho import set_path, save_tif, erase, count, count_branches, detect_and_color, gammaCorrection, calculate_area, find_contours, polygon, detect_features, name_to_number
import pandas as pd
import umap
import matplotlib.pyplot as plt
import hdbscan
import seaborn as sns
import numpy as np



# Load the data from the CSV file
i_path = "/Users/juanpablomayaarteaga/Desktop/Hilus/Prepro/"
o_path = set_path(i_path + "/Output_images/")
csv_path = set_path(o_path + "/Merged_Data/")
Plot_path = set_path(o_path + "Plots/")

data = pd.read_csv(csv_path + "Morphology.csv")






# Selecting features for UMAP
features_to_assess= ["sholl_crossing_processes", "Junctions", "polygon_eccentricities",
                     "cell_compactness", "soma_aspect_ratio", "cell_aspect_ratio",
                     "cell_solidity", "cell_perimeter",
                     "soma_eccentricity", "polygon_perimeters",  "cell_area", "Initial_Points", 
                     "ratio_branches", "Total_Branches_Length", "cell_circularity", 
                     "Branches",  "Circularity_soma", "Area_soma", "cell_eccentricity"
                     ]



# Extract the selected features from the dataset
selected_data = data[features_to_assess]


# Calculate correlation matrix
correlation_matrix = data[features_to_assess].corr()



######################################################################
######################################################################
######################################################################
######################################################################


n_neighbors=15

######################################################################
######################################################################
######################################################################
######################################################################

#################### WITHOUT CLUSTER


# Extract the selected features from the dataset
selected_data = data[features_to_assess]

# Calculate correlation matrix
correlation_matrix = data[features_to_assess].corr()

# Apply UMAP for dimensionality reduction
reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, random_state=24)
embedding = reducer.fit_transform(selected_data)

# Visualize the data with UMAP
plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.5)
plt.title('UMAP Analysis',  fontsize=18)
plt.xlabel('UMAP 1', fontsize=14 )
plt.ylabel('UMAP 2', fontsize=14 )
plt.grid(False)
plt.style.use('dark_background')

plt.savefig(Plot_path + f"UMAP_Analysis_{n_neighbors}.png", dpi=500)
plt.show()  # Show the plot

# Save the updated dataframe to a new CSV file
#data.to_csv(csv_path + "Morphology_UMAP.csv", index=False)



######################################################################
######################################################################
######################################################################
######################################################################

##################         WITH CLUSTER

######################################################################
######################################################################
######################################################################
######################################################################



# Set a fixed seed for NumPy
np.random.seed(24)
# Apply UMAP for dimensionality reduction
reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, random_state=24)
embedding = reducer.fit_transform(selected_data)
#15 y 0.1: 3 grupos
#6 y 0.1: 5 

# Apply HDBSCAN clustering to the UMAP-transformed data
clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=None, allow_single_cluster=True)
clusterer.fit(embedding)
labels = clusterer.fit_predict(embedding)
clusterer2 = hdbscan.HDBSCAN(min_cluster_size=5)
labels2 = clusterer2.fit_predict(embedding)


from sklearn.metrics import jaccard_score
jaccard_index = jaccard_score(labels, labels2, average='micro')

print("Jaccard Index:", jaccard_index)


# Define the cluster colors
cluster_colors = {
    0: (255, 0, 0),
    2: (0, 200, 0),
    1: (0, 200, 200),
    3: (200, 0, 200),
    4: (125, 125, 125)
    #5: (125, 125, 125)
    
}

# Convert cluster labels to colors
cluster_colors_array = np.array([cluster_colors.get(label, (0, 0, 0)) for label in clusterer.labels_])

# Visualize the clustered data with colors
plt.scatter(embedding[:, 0], embedding[:, 1], c=cluster_colors_array / 255.0, alpha=0.5)
plt.title('UMAP with HDBSCAN Clustering', fontsize=18)
plt.xlabel('UMAP 1', fontsize=14 )
plt.ylabel('UMAP 2', fontsize=14 )
plt.grid(False)
plt.style.use('dark_background')

# Add cluster labels to the plot
for cluster_label in np.unique(clusterer.labels_):
    if cluster_label != -1:  # Excluding noise points labeled as -1
        cluster_points = embedding[clusterer.labels_ == cluster_label]
        cluster_center = np.mean(cluster_points, axis=0)
        plt.text(cluster_center[0], cluster_center[1], str(cluster_label), fontsize=10, color='white')

plt.savefig(Plot_path + f"UMAP_HDBSCAN_{n_neighbors}.png", dpi=500)
plt.show()  # Show the plot

# Add a column to the original dataframe with cluster labels
data['UMAP_1'] = embedding[:, 0]
data['UMAP_2'] = embedding[:, 1]
data['Cluster_Labels'] = clusterer.labels_

# Save the updated dataframe to a new CSV file
data.to_csv(csv_path + f"Morphology_UMAP_HDBSCAN_{n_neighbors}.csv", index=False)



#mapper = umap.UMAP().fit(data)
import sklearn.datasets
import pandas as pd
import numpy as np
import umap
import umap.plot


######################################################################
######################################################################
######################################################################
######################################################################


connectivity_plot = umap.plot.connectivity(reducer, show_points=True, theme="inferno")
plt.savefig(Plot_path + f"connectivity_{n_neighbors}.png", dpi=800)
plt.show()


######################################################################
######################################################################
######################################################################
######################################################################


jacard_plot =umap.plot.diagnostic(reducer, diagnostic_type='neighborhood')

# Save the plot using plt.savefig
plt.savefig(Plot_path + f"neighborhood_Jacard_{n_neighbors}.png", dpi=800)

# Show the plot
plt.show()


######################################################################
######################################################################
######################################################################
######################################################################


######################################################################
######################################################################
######################################################################
######################################################################


######### INTERACTIVE
p = umap.plot.interactive(
    reducer,
    labels=data['Cluster_Labels'],
    hover_data=data[['Cell_ID']],
    color_key=cluster_colors,
    point_size=8
)
umap.plot.show(p)

######################################################################
######################################################################
######################################################################
######################################################################





######################################################################
######################################################################
######################################################################
######################################################################

#VIOLIN

######################################################################
######################################################################
######################################################################
######################################################################

import numpy as np

# Convert the keys of cluster_colors to strings and normalize the RGB values
cluster_colors_str = {str(key): np.array(value) / 255.0 for key, value in cluster_colors.items()}



"""

################## VIOLIN PLOTS REORDERED

# Specify the order of clusters for UMAP_1
umap1_cluster_order = ['3', '1', '0', '4']

# Plot the violin plot for UMAP_1 with cluster colors
plt.figure(figsize=(12, 8))
plt.style.use('dark_background')
sns.violinplot(x='Cluster_Labels', y='UMAP_1', data=data, palette=cluster_colors_str, order=umap1_cluster_order)
plt.title('UMAP 1 Distribution', fontsize=20, fontweight='bold', color='white')
plt.xlabel('Clusters', fontsize=18, fontweight='bold', color='white')
plt.ylabel('UMAP 1', fontsize=16, color='white')

plt.savefig(Plot_path + "UMAP_1_Violin_Plot_order.png", dpi=500)
plt.show()


# Specify the order of clusters for UMAP_2
umap2_cluster_order = ['0', '2', '3', '4', '1']

# Plot the violin plot for UMAP_2 with cluster colors
plt.figure(figsize=(12, 8))
plt.style.use('dark_background')
sns.violinplot(x='Cluster_Labels', y='UMAP_2', data=data, palette=cluster_colors_str, order=umap2_cluster_order)
plt.title('UMAP 2 Distribution', fontsize=20, fontweight='bold', color='white')
plt.xlabel('Clusters', fontsize=18, fontweight='bold', color='white')
plt.ylabel('UMAP 2', fontsize=16, color='white')

plt.savefig(Plot_path + "UMAP_2_Violin_Plot_order.png", dpi=500)
plt.show()


"""

######################################################################
######################################################################
######################################################################
######################################################################

#PIECHART

######################################################################
######################################################################
######################################################################
######################################################################

import matplotlib.pyplot as plt

cluster_colors = {
    2: "limegreen",
    1: "mediumturquoise",
    3: "darkorchid",
    0: "crimson",
    4: "gainsboro"
}

# Group by 'categories' and 'Cluster', then count occurrences
grouped = data.groupby(['categories', 'Cluster_Labels']).size().reset_index(name='Count')

category_order = ["VEH_SS", "CNEURO1_ESC", "VEH_ESC"]

# Group the data further by 'categories'
grouped_by_category = grouped.groupby('categories')

# Create subplots for each category to represent pie charts
fig, axes = plt.subplots(1, len(grouped_by_category), figsize=(16, 8), facecolor='black')  # Set the background color

# Plot pie charts for each category in the specified order
for ax, category in zip(axes, category_order):
    if category in grouped_by_category.groups:
        category_data = grouped_by_category.get_group(category)
        category_data = category_data.set_index('Cluster_Labels')['Count']
        colors = [cluster_colors.get(cluster, 'grey') for cluster in category_data.index]

        def make_autopct(values):
            def my_autopct(pct):
                total = sum(values)
                val = int(round(pct * total / 100.0))
                return '{p:.2f}%'.format(p=pct)
            return my_autopct

        ax.pie(category_data, labels=category_data.index, colors=colors, autopct=make_autopct(category_data.values), startangle=140)
        ax.set_title(f'{category}', fontweight='bold', color='white', fontsize=18)  # Set title color

        for text in ax.texts:
            text.set_fontweight('bold')
            text.set_color('white') # Set text color
            text.set_fontsize(11) 

# Save and display the plot
plt.tight_layout()
plt.savefig(Plot_path + f"Cluster_pie_chart_UMAP_{n_neighbors}.png", dpi=500, facecolor='black')  # Set the background color for the saved image
plt.show()








