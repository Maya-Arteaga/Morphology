#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 20:04:59 2023

@author: juanpablomayaarteaga
"""

import pandas as pd
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from morpho import set_path
import hdbscan
import sklearn.datasets
import pandas as pd
import numpy as np
import umap
import umap.plot
import time
import os

# Record start time
start_time = time.time()


# Load the data
i_path = "/Users/juanpablomayaarteaga/Desktop/Hilus/Prepro/"
o_path = set_path(i_path + "/Output_images/")
csv_path = set_path(o_path + "/Merged_Data/")
Plot_path = set_path(o_path + "Plots/")

data = pd.read_csv(csv_path + "Morphology.csv")


######################################################################
######################################################################
######################################################################
######################################################################


"""
############ RFE    0.4 test 
Selected Features: ['Circularity_soma', 'soma_compactness', 'soma_eccentricity', 'soma_aspect_ratio', 'Junctions', 'Initial_Points', 'ratio_branches', 'polygon_eccentricities', 'cell_compactness', 'cell_feret_diameter', 'cell_eccentricity', 'cell_aspect_ratio', 'cell_solidity', 'cell_convexity', 'sholl_num_circles']
Accuracy on Test Set: 0.31825273010920435
Classification Report:
                precision    recall  f1-score   support

CNEURO-01_ESC       0.38      0.55      0.45       150
  CNEURO1_ESC       0.28      0.32      0.30       142
   CNEURO1_SS       0.35      0.22      0.27       129
      VEH_ESC       0.25      0.25      0.25       107
       VEH_SS       0.30      0.19      0.24       113

     accuracy                           0.32       641
    macro avg       0.31      0.31      0.30       641
 weighted avg       0.31      0.32      0.31       641
 
 

Permutation Feature Importances: [ 0.0133995   0.00322581 -0.00645161  0.00322581 -0.00223325 -0.00223325
  0.00372208  0.0101737   0.01927213  0.01612903 -0.00620347  0.00306038
  0.01943755  0.00057899  0.00041356  0.00471464 -0.00090984  0.00645161
  0.00115798  0.00413565  0.00181969  0.02051282  0.00587262 -0.00934657
  0.00033085  0.01100083  0.00082713 -0.00504549  0.00330852 -0.00289495]
Selected Features: ['Area_soma', 'End_Points', 'Junctions', 'Branches', 'ratio_branches', 'cell_compactness', 'cell_solidity']
Cross-Validation Scores: [0.32978723 0.30319149 0.34574468 0.34224599 0.34759358]
Accuracy on Test Set (Selected Features): 0.33002481389578164
Classification Report (Selected Features):
                precision    recall  f1-score   support

CNEURO-01_ESC       0.40      0.49      0.44        76
  CNEURO1_ESC       0.26      0.23      0.24        62
   CNEURO1_SS       0.36      0.29      0.32        93
      VEH_ESC       0.28      0.32      0.30        82
       VEH_SS       0.33      0.32      0.32        90

     accuracy                           0.33       403
    macro avg       0.33      0.33      0.32       403
 weighted avg       0.33      0.33      0.33       403
 
 
Selected Features: ['Area_soma', 'Perimeter_soma', 'Circularity_soma', 'soma_compactness', 'soma_feret_diameter', 'soma_aspect_ratio', 'End_Points', 'Junctions', 'Branches', 'Initial_Points', 'ratio_branches', 'polygon_area', 'polygon_compactness', 'polygon_eccentricities', 'cell_area', 'cell_perimeter', 'cell_compactness', 'cell_aspect_ratio', 'cell_solidity', 'cell_convexity', 'sholl_max_distance']
Accuracy on Test Set (Selected Features): 0.3407821229050279
Classification Report (Selected Features):
                precision    recall  f1-score   support

CNEURO-01_ESC       0.44      0.49      0.46        99
  CNEURO1_ESC       0.34      0.23      0.28        91
   CNEURO1_SS       0.34      0.31      0.32       129
      VEH_ESC       0.29      0.35      0.32       104
       VEH_SS       0.31      0.32      0.31       114

     accuracy                           0.34       537
    macro avg       0.34      0.34      0.34       537
 weighted avg       0.34      0.34      0.34       537

Selected Features: ['Circularity_soma', 'soma_compactness', 'soma_eccentricity', 'soma_aspect_ratio', 'Junctions', 'Initial_Points', 'ratio_branches', 'polygon_eccentricities', 'polygon_feret_diameters', 'cell_compactness', 'cell_feret_diameter', 'cell_eccentricity', 'cell_aspect_ratio', 'cell_solidity', 'cell_convexity']
Accuracy on Test Set: 0.30855018587360594
Classification Report:
                precision    recall  f1-score   support

CNEURO-01_ESC       0.45      0.58      0.51        50
  CNEURO1_ESC       0.20      0.11      0.14        44
   CNEURO1_SS       0.27      0.41      0.32        58
      VEH_ESC       0.27      0.46      0.34        48
       VEH_SS       0.43      0.04      0.08        69

     accuracy                           0.31       269
    macro avg       0.32      0.32      0.28       269
 weighted avg       0.33      0.31      0.27       269




##### Permutation and random FOREST
features_to_assess = ['Circularity_soma', 'soma_compactness', 'soma_eccentricity', 
                      'soma_aspect_ratio', 'Junctions', 'Initial_Points', 
                      'ratio_branches', 'polygon_eccentricities', 'cell_compactness', 
                      'cell_feret_diameter', 'cell_eccentricity', 'cell_aspect_ratio', 
                      'cell_solidity', 'cell_convexity', 'sholl_num_circles']


Selected Features: ['Circularity_soma', 'soma_compactness', 'soma_eccentricity', 
                    'soma_aspect_ratio', 'Junctions', 'Initial_Points', 
                    'ratio_branches', 'polygon_eccentricities', 'cell_compactness',
                    'cell_feret_diameter', 'cell_eccentricity', 'cell_aspect_ratio', 
                    'cell_solidity', 'cell_convexity', 'sholl_num_circles']

Accuracy on Test Set: 0.3073322932917317
Classification Report:
                precision    recall  f1-score   support

CNEURO-01_ESC       0.35      0.53      0.42       150
  CNEURO1_ESC       0.35      0.36      0.36       142
   CNEURO1_SS       0.24      0.19      0.21       129
      VEH_ESC       0.16      0.14      0.15       107
       VEH_SS       0.35      0.25      0.29       113

     accuracy                           0.31       641
    macro avg       0.29      0.29      0.29       641
 weighted avg       0.30      0.31      0.30       641
"""




features_to_assess = ['Circularity_soma', 'soma_compactness', 'soma_eccentricity', 
                    'soma_aspect_ratio', 'Junctions', 'Initial_Points', 
                    'ratio_branches', 'polygon_eccentricities', 'cell_compactness', 
                    'cell_feret_diameter', 'cell_eccentricity', 'cell_aspect_ratio', 
                    'cell_solidity', 'cell_convexity', 'sholl_num_circles']






# Extract the selected features from the dataset
selected_data = data[features_to_assess]



######################################################################
######################################################################
######################################################################
######################################################################

#######################   CORRELATION MATRIX   ########################


######################################################################
######################################################################
######################################################################
######################################################################


correlation_matrix = selected_data.corr()

# Set up the matplotlib figure
plt.figure(figsize=(12, 10))

# Create a heatmap with a color map
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')

# Add a title
plt.title('Correlation Matrix Heatmap')
plt.savefig(Plot_path + "Correlation_Matrix_RFE_SELECTED.png", dpi=800, bbox_inches="tight")

# Show the plot
plt.show()


######################################################################
######################################################################
######################################################################
######################################################################

#############################   UMAP   ###############################


######################################################################
######################################################################
######################################################################
######################################################################



# Apply UMAP for dimensionality reduction using the PCA result
n_neighbors = 10
min_dist= 0.01


reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=24)
embedding = reducer.fit_transform(selected_data)
#embedding = reducer.fit_transform(selected_pca_result)

# Visualize the data with UMAP
plt.style.use('dark_background')
plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.5)
plt.title(f'UMAP (n={n_neighbors}, d={min_dist})', fontsize=18)
plt.xlabel('UMAP 1', fontsize=14)
plt.ylabel('UMAP 2', fontsize=14)
plt.grid(False)
plt.style.use('dark_background')

plt.savefig(Plot_path + f"UMAP_{n_neighbors}_{min_dist}.png", dpi=500)
plt.show()  # Show the plot

# Save the updated dataframe to a new CSV file
# data.to_csv(csv_path + "Morphology_UMAP.csv", index=False)





######################################################################
######################################################################
######################################################################
######################################################################

############################   HDBSCAN    #############################


######################################################################
######################################################################
######################################################################
######################################################################

#n_neighbors = 15

# Apply UMAP for dimensionality reduction
reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=24)
embedding = reducer.fit_transform(selected_data)


# Apply HDBSCAN clustering to the UMAP-transformed data
clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=15, allow_single_cluster=True)
clusterer.fit(embedding)
labels = clusterer.fit_predict(embedding)
#clusterer2 = hdbscan.HDBSCAN(min_cluster_size=5)
#labels2 = clusterer2.fit_predict(embedding)


#from sklearn.metrics import jaccard_score
#jaccard_index = jaccard_score(labels, labels2, average='micro')

#print("Jaccard Index:", jaccard_index)


# Define the cluster colors RGB
# Custom color palette
cluster_colors = {
    0: "orangered",
    1: "crimson",
    4: "paleturquoise",
    2: "gold",
    3: "limegreen",
    5: "mediumturquoise",
    #6: "lime" darkorchid
    #seagreen darkorchid

    
}

# Convert cluster labels to colors
#cluster_colors_array = np.array([cluster_colors.get(label, (0, 0, 0)) for label in clusterer.labels_])
default_color = "gray"
cluster_colors_array = np.array([cluster_colors.get(label, default_color) for label in clusterer.labels_])


# Visualize the clustered data with colors
#plt.scatter(embedding[:, 0], embedding[:, 1], c=cluster_colors_array / 255.0, alpha=0.5)
plt.scatter(embedding[:, 0], embedding[:, 1], c=cluster_colors_array, alpha=0.3)



plt.title(f'UMAP with HDBSCAN Clustering', fontsize=18)
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

plt.savefig(Plot_path + f"UMAP_HDBSCAN_{n_neighbors}_{min_dist}.png", dpi=500)
plt.show()  # Show the plot

# Add a column to the original dataframe with cluster labels
data['UMAP_1'] = embedding[:, 0]
data['UMAP_2'] = embedding[:, 1]
data['Cluster_Labels'] = clusterer.labels_

# Save the updated dataframe to a new CSV file
data.to_csv(csv_path + f"Morphology_PCA_UMAP_HDBSCAN_{n_neighbors}.csv", index=False)




######################################################################
######################################################################
######################################################################
######################################################################

###########################   PIE CHART   ############################


######################################################################
######################################################################
######################################################################
######################################################################




import matplotlib.pyplot as plt

# Custom color palette
cluster_colors = cluster_colors

# Group by 'categories' and 'Cluster', then count occurrences
grouped = data.groupby(['categories', 'Cluster_Labels']).size().reset_index(name='Count')

category_order = ["VEH_SS",  "VEH_ESC", "CNEURO-01_ESC", "CNEURO1_ESC", "CNEURO1_SS"]
category_labels = {"VEH_SS": "VEH SS", "CNEURO1_ESC": "CNEURO 1.0", "VEH_ESC": "ESC", "CNEURO-01_ESC": "CNEURO 0.1", "CNEURO1_SS": "CNEURO 1.0 SS"}

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

        ax.set_xticklabels([category_labels.get(label, label) for label in category_order])
        ax.pie(category_data, labels=category_data.index, colors=colors, autopct=make_autopct(category_data.values), startangle=140)
        ax.set_title(f'{category_labels.get(category, category)}', fontweight='bold', color='white', fontsize=18)  # Use custom label for the title

        for text in ax.texts:
            text.set_fontweight('bold')
            text.set_color('white')  # Set text color
            text.set_fontsize(11) 

# Save and display the plot
plt.tight_layout()
plt.savefig(Plot_path + f"Pie_chart_Clusters_{n_neighbors}_{min_dist}.png", dpi=500, facecolor='black')  # Set the background color for the saved image
plt.show()







######################################################################
######################################################################
######################################################################
######################################################################

#######################   HISTOGRAM CLUSTERS   ########################


######################################################################
######################################################################
######################################################################
######################################################################



import seaborn as sns
import matplotlib.pyplot as plt


# Filter out rows with Cluster_Labels equal to -1
filtered_data = data[data['Cluster_Labels'] != -1]

# Set the style for the plots (optional)
sns.set(style="whitegrid")

# Create a count plot using seaborn with a custom color palette
plt.figure(figsize=(12, 8))  # Adjust the figure size if needed

# Set the background color to black
plt.style.use('dark_background')

# Custom color palette
cluster_colors = {
    0: "orangered",
    1: "crimson",
    4: "paleturquoise",
    2: "gold",
    3: "limegreen",
    5: "mediumturquoise",
}

# Specify the order of categories and corresponding labels
category_order = ["VEH_SS",  "VEH_ESC", "CNEURO-01_ESC", "CNEURO1_ESC", "CNEURO1_SS"]
category_labels = {"VEH_SS": "VEH SS", "CNEURO1_ESC": "CNEURO 1.0", "VEH_ESC": "ESC", "CNEURO-01_ESC": "CNEURO 0.1", "CNEURO1_SS": "CNEURO 1.0 SS"}

# Use the custom palette and order in the countplot
ax = sns.countplot(x="categories", hue="Cluster_Labels", data=filtered_data, palette=cluster_colors, order=category_order)

# Customize x-axis labels
ax.set_xticklabels([category_labels.get(label, label) for label in category_order])

# Add title and labels
plt.title("Count of Cells for Each Cluster")
plt.xlabel("Categories")
plt.ylabel("Count of Cells")
plt.style.use('dark_background')

# Add legend outside the plot area
ax.legend(title='Clusters', loc='upper left', labels=[f'Cluster {i}' for i in cluster_colors], bbox_to_anchor=(1, 1))

# Automatically adjust subplot parameters for better spacing
plt.tight_layout()

plt.savefig(Plot_path + f"Histogram_Clusters_{n_neighbors}_{min_dist}.png", dpi=800, bbox_inches="tight")

# Show the plot
plt.show()















import seaborn as sns
import matplotlib.pyplot as plt

# Filter out rows with Cluster_Labels equal to -1
filtered_data = data[data['Cluster_Labels'] != -1]

# Set the style for the plots (optional)
sns.set(style="whitegrid")

# Create a count plot using seaborn with a custom color palette
plt.figure(figsize=(12, 8))  # Adjust the figure size if needed

# Set the background color to black
plt.style.use('dark_background')

# Specify the order of categories and corresponding labels
category_order = ["VEH_SS",  "VEH_ESC", "CNEURO-01_ESC", "CNEURO1_ESC", "CNEURO1_SS"]
category_labels = {"VEH_SS": "VEH SS", "CNEURO1_ESC": "CNEURO 1.0 ESC", "VEH_ESC": "ESC", "CNEURO-01_ESC": "CNEURO 0.1 ESC", "CNEURO1_SS": "CNEURO 1.0 SS"}

# Specify the order of clusters
cluster_order = [3,5,4,2,0,1]

# Use the custom palette and order in the countplot
ax = sns.countplot(x="categories", hue="Cluster_Labels", data=filtered_data, palette=cluster_colors, order=category_order, hue_order=cluster_order)

# Customize x-axis labels
ax.set_xticklabels([category_labels.get(label, label) for label in category_order])

# Add title and labels
plt.title("Count of Cells for Each Cluster")
plt.xlabel("Categories")
plt.ylabel("Count of Cells")
plt.style.use('dark_background')

# Add legend
ax.legend(title='Clusters', loc='upper right', labels=[f'Cluster {i}' for i in cluster_order])

plt.savefig(Plot_path + f"Histogram_Clusters_{n_neighbors}_{min_dist}_order.png", dpi=800, bbox_inches="tight")

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
########################    NOISE   ##################################
######################################################################
######################################################################


# Define the cluster colors RGB
# Custom color palette
cluster_colors = {
    -1: "crimson",
    0: "gainsboro",
    1: "gainsboro",
    3: "gainsboro",
    2: "gainsboro",
    4: "gainsboro",
    5: "gainsboro",
    6: "gainsboro"

    
}

# Convert cluster labels to colors
#cluster_colors_array = np.array([cluster_colors.get(label, (0, 0, 0)) for label in clusterer.labels_])
default_color = "gray"
cluster_colors_array = np.array([cluster_colors.get(label, default_color) for label in clusterer.labels_])


# Visualize the clustered data with colors
#plt.scatter(embedding[:, 0], embedding[:, 1], c=cluster_colors_array / 255.0, alpha=0.5)
plt.scatter(embedding[:, 0], embedding[:, 1], c=cluster_colors_array, alpha=0.5)



plt.title(f'UMAP-HDBSCAN: NOISE (n ={n_neighbors})', fontsize=18)
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

plt.savefig(Plot_path + f"UMAP_HDBSCAN_{n_neighbors}_{min_dist}_noise.png", dpi=500)
plt.show()  # Show the plot




######################################################################
######################################################################
######################################################################
######################################################################

#######################   SEPARATED BY GROUP   #######################


######################################################################
######################################################################
######################################################################
######################################################################








import matplotlib.pyplot as plt
import umap

# Assuming data is your DataFrame, and embedding is the result of UMAP
#n_neighbors = 15
reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=24)
embedding = reducer.fit_transform(selected_data)  

cluster_colors = {
    "VEH_SS": "limegreen",
    "CNEURO1_ESC": "yellow",
    "VEH_ESC": "crimson",
    "CNEURO-01_ESC": "darkorange",
    "CNEURO1_SS": "mediumturquoise"
    
    
}

# Iterate over categories and plot points with respective colors
plt.figure(figsize=(12, 8))
plt.style.use('dark_background')

for category, color in cluster_colors.items():
    category_data = data[data['categories'] == category]
    plt.scatter(
        embedding[category_data.index, 0],
        embedding[category_data.index, 1],
        color=color,
        alpha=0.5,
        label=category
    )

plt.title(f'UMAP Analysis by Group (n_neighbors={n_neighbors})', fontsize=18)
plt.xlabel('UMAP 1', fontsize=14)
plt.ylabel('UMAP 2', fontsize=14)
plt.legend()
plt.grid(False)

plt.savefig(Plot_path + f"UMAP_{n_neighbors}_{min_dist}_Categories.png", dpi=800)
plt.show()




######################################################################
######################################################################
#######################      VEH_SS     ##############################
######################################################################
######################################################################


cluster_colors = {
    "VEH_SS": "limegreen",
    "CNEURO1_ESC": "gainsboro",
    "VEH_ESC": "gainsboro",
    "CNEURO-01_ESC": "gainsboro",
    "CNEURO1_SS": "gainsboro"
    
    
}

# Iterate over categories and plot points with respective colors
plt.figure(figsize=(12, 8))
plt.style.use('dark_background')

for category, color in cluster_colors.items():
    category_data = data[data['categories'] == category]
    plt.scatter(
        embedding[category_data.index, 0],
        embedding[category_data.index, 1],
        color=color,
        alpha=1.0,
        label=category
    )

plt.title('UMAP Analysis by Group: VEH SS', fontsize=18)
plt.xlabel('UMAP 1', fontsize=14)
plt.ylabel('UMAP 2', fontsize=14)
plt.legend()
plt.grid(False)

plt.savefig(Plot_path + f"UMAP_{n_neighbors}_{min_dist}_Categories_VEH_SS.png", dpi=800)
plt.show()



######################################################################
######################################################################
#######################      CNEURO1_ESC     ##############################
######################################################################
######################################################################


cluster_colors = {
    "VEH_SS": "gainsboro",
    "CNEURO1_ESC": "gold",
    "VEH_ESC": "gainsboro",
    "CNEURO-01_ESC": "gainsboro",
    "CNEURO1_SS": "gainsboro"
    
    
}

# Iterate over categories and plot points with respective colors
plt.figure(figsize=(12, 8))
plt.style.use('dark_background')

for category, color in cluster_colors.items():
    category_data = data[data['categories'] == category]
    plt.scatter(
        embedding[category_data.index, 0],
        embedding[category_data.index, 1],
        color=color,
        alpha=1.0,
        label=category
    )

plt.title('UMAP Analysis by Group: CNEURO 1.0 ESC', fontsize=18)
plt.xlabel('UMAP 1', fontsize=14)
plt.ylabel('UMAP 2', fontsize=14)
plt.legend()
plt.grid(False)

plt.savefig(Plot_path + f"UMAP_{n_neighbors}_{min_dist}_Categories_CNEURO1_ESC .png", dpi=800)
plt.show()


######################################################################
######################################################################
#######################      VEH_ESC     ##############################
######################################################################
######################################################################


cluster_colors = {
    "VEH_SS": "gainsboro",
    "CNEURO1_ESC": "gainsboro",
    "VEH_ESC": "crimson",
    "CNEURO-01_ESC": "gainsboro",
    "CNEURO1_SS": "gainsboro"
    
    
}

# Iterate over categories and plot points with respective colors
plt.figure(figsize=(12, 8))
plt.style.use('dark_background')

for category, color in cluster_colors.items():
    category_data = data[data['categories'] == category]
    plt.scatter(
        embedding[category_data.index, 0],
        embedding[category_data.index, 1],
        color=color,
        alpha=1.0,
        label=category
    )

plt.title('UMAP Analysis by Group: VEH ESC', fontsize=18)
plt.xlabel('UMAP 1', fontsize=14)
plt.ylabel('UMAP 2', fontsize=14)
plt.legend()
plt.grid(False)

plt.savefig(Plot_path + f"UMAP_{n_neighbors}_{min_dist}_Categories_VEH_ESC.png", dpi=800)
plt.show()







######################################################################
######################################################################
#######################      CNEURO-01_ESC     ##############################
######################################################################
######################################################################


cluster_colors = {
    "VEH_SS": "gainsboro",
    "CNEURO1_ESC": "gainsboro",
    "VEH_ESC": "gainsboro",
    "CNEURO-01_ESC": "darkorange",
    "CNEURO1_SS": "gainsboro"
    
    
}

# Iterate over categories and plot points with respective colors
plt.figure(figsize=(12, 8))
plt.style.use('dark_background')

for category, color in cluster_colors.items():
    category_data = data[data['categories'] == category]
    plt.scatter(
        embedding[category_data.index, 0],
        embedding[category_data.index, 1],
        color=color,
        alpha=1.0,
        label=category
    )

plt.title('UMAP Analysis by Group: CNEURO 0.1 ESC', fontsize=18)
plt.xlabel('UMAP 1', fontsize=14)
plt.ylabel('UMAP 2', fontsize=14)
plt.legend()
plt.grid(False)

plt.savefig(Plot_path + f"UMAP_{n_neighbors}_{min_dist}_Categories_CNEURO-01_ESC.png", dpi=800)
plt.show()








######################################################################
######################################################################
#######################      CNEURO1_SS     ##############################
######################################################################
######################################################################


cluster_colors = {
    "VEH_SS": "gainsboro",
    "CNEURO1_ESC": "gainsboro",
    "VEH_ESC": "gainsboro",
    "CNEURO-01_ESC": "gainsboro",
    "CNEURO1_SS": "mediumturquoise"
    
    
}

# Iterate over categories and plot points with respective colors
plt.figure(figsize=(12, 8))
plt.style.use('dark_background')

for category, color in cluster_colors.items():
    category_data = data[data['categories'] == category]
    plt.scatter(
        embedding[category_data.index, 0],
        embedding[category_data.index, 1],
        color=color,
        alpha=1.0,
        label=category
    )

plt.title('UMAP Analysis by Group: CNEURO 1.0 SS', fontsize=18)
plt.xlabel('UMAP 1', fontsize=14)
plt.ylabel('UMAP 2', fontsize=14)
plt.legend()
plt.grid(False)

plt.savefig(Plot_path + f"UMAP_{n_neighbors}_{min_dist}_Categories_CNEURO1_SS.png", dpi=800)
plt.show()



######################################################################
######################################################################
######################################################################
######################################################################




######### JACCAR INDEX

jacard_plot =umap.plot.diagnostic(reducer, diagnostic_type='neighborhood')

# Save the plot using plt.savefig
plt.savefig(Plot_path + f"Neighborhood_Jacard_{n_neighbors}_{min_dist}.png", dpi=800)

# Show the plot
plt.show()


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

######### Local Dimension

local_dims = umap.plot.diagnostic(reducer, diagnostic_type='local_dim')
# Save the plot using plt.savefig
plt.savefig(Plot_path + f"Local_Dim_{n_neighbors}_{min_dist}.png", dpi=800)

# Show the plot
plt.show()




umap.plot.diagnostic(reducer, diagnostic_type='vq')
plt.savefig(Plot_path + f"Vector_Quantization_{n_neighbors}_{min_dist}.png", dpi=800)

# Show the plot
plt.show()

######################################################################
######################################################################
######################################################################
######################################################################

######### CONNECTIVITY


connectivity_plot = umap.plot.connectivity(reducer, show_points=True, theme="inferno")
plt.savefig(Plot_path + f"Connectivity_{n_neighbors}_{min_dist}_points.png", dpi=1000)
plt.show()


umap.plot.connectivity(reducer, edge_bundling='hammer', theme="inferno")
plt.savefig(Plot_path + f"Connectivity_{n_neighbors}_{min_dist}_hammer.png", dpi=1000)
plt.show()





#####################   ANOVA & TUKEY  ######################


######################################################################
######################################################################
######################################################################
######################################################################




import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Filter out rows with Cluster_Labels equal to -1
filtered_data = data[data['Cluster_Labels'] != -1]

# Perform ANOVA test
formula = 'Cluster_Labels ~ categories'
model = ols(formula, data=filtered_data).fit()
anova_result = sm.stats.anova_lm(model, typ=2)

# Print ANOVA results
print("ANOVA Results:")
print(anova_result)

print("...")
print("...")
print("...")

# Filter out rows with Cluster_Labels equal to -1
filtered_data = data[data['Cluster_Labels'] != -1]

# Perform Tukey's HSD test
tukey_result = pairwise_tukeyhsd(filtered_data['Cluster_Labels'], filtered_data['categories'])

# Print Tukey's HSD results
print("\nTukey's HSD Results:")
print(tukey_result)


print("...")
print("...")
print("...")

#Dataframe
tukey_df = pd.DataFrame(tukey_result._results_table.data[1:], columns=tukey_result._results_table.data[0])
print(tukey_df)

print("...")
print("...")
print("...")






######################################################################
######################################################################
######################################################################
######################################################################


#####################   Kruskal-Wallis & DUNN   ######################


######################################################################
######################################################################
######################################################################
######################################################################

import pandas as pd
from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn
from statsmodels.stats import multitest


"""
# Filter out rows with Cluster_Labels equal to -1
filtered_data = data[data['Cluster_Labels'] != -1]

# Perform Kruskal-Wallis test
kruskal_result = kruskal(*[group['Cluster_Labels'] for name, group in filtered_data.groupby('categories')])
"""
kruskal_result = kruskal(*[group['Cluster_Labels'] for name, group in data.groupby('categories')])


print("...")
print("...")
print("...")

# Print Kruskal-Wallis result
print("Kruskal-Wallis Results:")
print(kruskal_result)

print("...")
print("...")
print("...")

# Perform Dunn's test for pairwise comparisons
dunn_result = posthoc_dunn(data, val_col='Cluster_Labels', group_col='categories')

# Print Dunn's test results
print("\nDunn's Test Results:")
print(dunn_result)

# Apply Bonferroni correction
bonferroni_corrected = multitest.multipletests(dunn_result.values.flatten(), method='bonferroni')
dunn_result_bonferroni = pd.DataFrame(bonferroni_corrected[1].reshape(dunn_result.shape), index=dunn_result.index, columns=dunn_result.columns)

# Apply False Discovery Rate (FDR) correction
fdr_corrected = multitest.multipletests(dunn_result.values.flatten(), method='fdr_bh')
dunn_result_fdr = pd.DataFrame(fdr_corrected[1].reshape(dunn_result.shape), index=dunn_result.index, columns=dunn_result.columns)

# Print corrected results
print("\nDunn's Test Results (Bonferroni Corrected):")
print(dunn_result_bonferroni)


print("...")
print("...")
print("...")


# Convert Kruskal-Wallis result to DataFrame
kruskal_result_df = pd.DataFrame({"Kruskal-Wallis": [kruskal_result.statistic], "p-value": [kruskal_result.pvalue]})

# Convert Dunn's test results to DataFrames
dunn_result_df = pd.DataFrame(dunn_result)
dunn_result_bonferroni_df = pd.DataFrame(dunn_result_bonferroni)
dunn_result_fdr_df = pd.DataFrame(dunn_result_fdr)


print("\nDunn's Test Results (FDR Corrected):")
print(dunn_result_fdr)


# Function to apply color-coding based on the specified conditions
def apply_color(x):
    if x <= 0.001:
        return 'color: red'
    elif 0.001 < x <= 0.01:
        return 'color: purple'
    elif 0.01 < x <= 0.05:
        return 'color: blue'
    else:
        return ''

# Save results to HTML file at the specified path
html_file_path = os.path.join(csv_path, 'Clusters_results.html')

with open(html_file_path, 'w') as f:
    f.write("<html>\n<head>\n<title>Kruskal-Wallis & Dunn Results</title>\n</head>\n<body>\n")

    # Add Kruskal-Wallis results to HTML
    f.write("<h2>Kruskal-Wallis Results:</h2>\n")
    f.write(kruskal_result_df.style.applymap(apply_color).to_html(escape=False))

    # Add Dunn's test results to HTML
    f.write("<h2>Dunn's Test Results:</h2>\n")
    f.write(dunn_result_df.style.applymap(apply_color).to_html(escape=False))

    # Add Bonferroni-corrected Dunn's test results to HTML
    f.write("<h2>Dunn's Test Results (Bonferroni Corrected):</h2>\n")
    f.write(dunn_result_bonferroni_df.style.applymap(apply_color).to_html(escape=False))

    # Add FDR-corrected Dunn's test results to HTML
    f.write("<h2>Dunn's Test Results (FDR Corrected):</h2>\n")
    f.write(dunn_result_fdr_df.style.applymap(apply_color).to_html(escape=False))

    f.write("</body>\n</html>")

print(f"Results saved at: {html_file_path}")







# Save results to CSV file
csv_file_path = os.path.join(csv_path, 'Clusters_results.csv')

# Create a DataFrame for the results
results_df = pd.DataFrame({
    'Kruskal-Wallis Statistic': [kruskal_result.statistic],
    'Kruskal-Wallis p-value': [kruskal_result.pvalue],
})

# Add Dunn's test results to the DataFrame
results_df = pd.concat([results_df, dunn_result_df.add_suffix(' (Dunn Test)')], axis=1)


# Add Bonferroni-corrected Dunn's test results to the DataFrame
results_df_bonferroni = dunn_result_bonferroni_df.copy()
results_df_bonferroni.columns = results_df_bonferroni.columns + ' (Bonferroni Corrected)'
results_df = pd.concat([results_df, results_df_bonferroni], axis=1)

# Add FDR-corrected Dunn's test results to the DataFrame
results_df_fdr = dunn_result_fdr_df.copy()
results_df_fdr.columns = results_df_fdr.columns + ' (FDR Corrected)'
results_df = pd.concat([results_df, results_df_fdr], axis=1)

# Save the DataFrame to CSV
results_df.to_csv(csv_file_path, index=True)


print(f"Results saved at: {csv_file_path}")


# Record end time
end_time = time.time()

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time} seconds")



