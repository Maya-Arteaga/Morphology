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

#DECISSION TREE 
features_to_assess= ["polygon_eccentricities", "sholl_crossing_processes", "Branches",
                     
                     "cell_solidity", "soma_eccentricity", "cell_compactness",
                     
                     "cell_area", "soma_aspect_ratio", "cell_convexity",
                     
                     "End_Points", "polygon_perimeters",  "polygon_area",
                     
                     "Total_Branches_Length"
                     ]


"""
"""
#RFE results
Selected Features: ['Circularity_soma', 'soma_compactness', 'soma_eccentricity', 'End_Points', 'Junctions', 'Branches', 'ratio_branches', 'polygon_eccentricities', 'polygon_feret_diameters', 'cell_compactness', 'cell_feret_diameter', 'cell_eccentricity', 'cell_aspect_ratio', 'cell_convexity', 'sholl_num_circles']
Accuracy on Test Set: 0.5454545454545454
Classification Report:
               precision    recall  f1-score   support

 CNEURO1_ESC       0.57      0.52      0.55        23
     VEH_ESC       0.58      0.88      0.70         8
      VEH_SS       0.45      0.38      0.42        13

    accuracy                           0.55        44
   macro avg       0.54      0.59      0.55        44
weighted avg       0.54      0.55      0.54        44
"""

# Select features for UMAP
features_to_assess = ['Circularity_soma', 'soma_compactness', 'soma_eccentricity',
                      'End_Points', 'Junctions', 'Branches', 'ratio_branches',
                      'polygon_eccentricities', 'polygon_feret_diameters',
                      'cell_compactness', 'cell_feret_diameter', 'cell_eccentricity',
                      'cell_aspect_ratio', 'cell_convexity', 'sholl_num_circles']







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
plt.savefig(Plot_path + "Correlation_Matrix_SELECTED.png", dpi=800)

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



# Perform PCA
pca = PCA(n_components=5) 
pca_result = pca.fit_transform(selected_data)



"""


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data_standardized = scaler.fit_transform(selected_data)

# Perform PCA
pca = PCA()
pca_result = pca.fit_transform(data_standardized)

# Calculate the cumulative explained variance
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

# Determine the number of components to retain for 90% variance
num_components = np.argmax(cumulative_variance_ratio >= 0.9) + 1

# Retain the selected number of components
selected_pca_result = pca_result[:, :num_components]
"""
# Apply UMAP for dimensionality reduction using the PCA result
n_neighbors = 15


reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, random_state=24)
embedding = reducer.fit_transform(pca_result)
#embedding = reducer.fit_transform(selected_pca_result)

# Visualize the data with UMAP
plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.5)
plt.title(f'UMAP (n_neighbors={n_neighbors})', fontsize=18)
plt.xlabel('UMAP 1', fontsize=14)
plt.ylabel('UMAP 2', fontsize=14)
plt.grid(False)
plt.style.use('dark_background')

plt.savefig(Plot_path + f"UMAP_Analysis_with_PCA_{n_neighbors}.png", dpi=500)
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
reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, random_state=24)
embedding = reducer.fit_transform(pca_result)
#15 y 0.1: 3 grupos
#6 y 0.1: 5 

# Apply HDBSCAN clustering to the UMAP-transformed data
clusterer = hdbscan.HDBSCAN(min_cluster_size=6, min_samples=7, allow_single_cluster=True)
clusterer.fit(embedding)
labels = clusterer.fit_predict(embedding)
#clusterer2 = hdbscan.HDBSCAN(min_cluster_size=5)
#labels2 = clusterer2.fit_predict(embedding)


#from sklearn.metrics import jaccard_score
#jaccard_index = jaccard_score(labels, labels2, average='micro')

#print("Jaccard Index:", jaccard_index)


# Define the cluster colors
cluster_colors = {
    -1: (255, 255, 255),
    3: (255, 0, 0),
    0: (0, 200, 0),
    1: (0, 200, 200),
    2: (200, 0, 200),
    4: (0, 125, 125),
    5: (125, 125, 125)
    
}

# Convert cluster labels to colors
cluster_colors_array = np.array([cluster_colors.get(label, (0, 0, 0)) for label in clusterer.labels_])

# Visualize the clustered data with colors
plt.scatter(embedding[:, 0], embedding[:, 1], c=cluster_colors_array / 255.0, alpha=0.5)
plt.title(f'UMAP with HDBSCAN Clustering (n_neighbors={n_neighbors})', fontsize=18)
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

plt.savefig(Plot_path + f"RFE_PCA_UMAP_HDBSCAN_{n_neighbors}.png", dpi=500)
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

#######################   SEPARATED BY GROUP   #######################


######################################################################
######################################################################
######################################################################
######################################################################








import matplotlib.pyplot as plt
import umap

# Assuming data is your DataFrame, and embedding is the result of UMAP
#n_neighbors = 15
reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, random_state=24)
embedding = reducer.fit_transform(pca_result)  # Assuming pca_result is defined

# Define cluster colors
cluster_colors = {
    "VEH_SS": "limegreen",
    "CNEURO1_ESC": "mediumturquoise",
    "VEH_ESC": "crimson"
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

plt.title(f'UMAP Analysis with PCA (n_neighbors={n_neighbors})', fontsize=18)
plt.xlabel('UMAP 1', fontsize=14)
plt.ylabel('UMAP 2', fontsize=14)
plt.legend()
plt.grid(False)

plt.savefig(Plot_path + f"UMAP_Analysis_with_PCA_{n_neighbors}_categories.png", dpi=500)
plt.show()



######################################################################
######################################################################
######################################################################
######################################################################

import sklearn.datasets
import pandas as pd
import numpy as np
import umap
import umap.plot


######### JACCAR INDEX

jacard_plot =umap.plot.diagnostic(reducer, diagnostic_type='neighborhood')

# Save the plot using plt.savefig
plt.savefig(Plot_path + f"neighborhood_Jacard_{n_neighbors}.png", dpi=800)

# Show the plot
plt.show()


######################################################################
######################################################################
######################################################################
######################################################################

######### CONNECTIVITY


connectivity_plot = umap.plot.connectivity(reducer, show_points=True, theme="inferno")
plt.savefig(Plot_path + f"connectivity_{n_neighbors}.png", dpi=800)
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

###########################   PIE CHART   ############################


######################################################################
######################################################################
######################################################################
######################################################################




import matplotlib.pyplot as plt

cluster_colors = {
    0: "limegreen",
    1: "mediumturquoise",
    2: "darkorchid",
    3: "crimson",
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
plt.savefig(Plot_path + f"Cluster_pie_chart_PCA_UMAP_{n_neighbors}.png", dpi=500, facecolor='black')  # Set the background color for the saved image
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
    0: "limegreen",
    1: "mediumturquoise",
    2: "darkorchid",
    3: "crimson",
    #4: "gainsboro"
}

# Specify the order of categories and corresponding labels
category_order = ["VEH_SS", "CNEURO1_ESC", "VEH_ESC"]
category_labels = {"VEH_SS": "VEH SS", "CNEURO1_ESC": "CNEURO1", "VEH_ESC": "ESC"}

# Use the custom palette and order in the countplot
ax = sns.countplot(x="categories", hue="Cluster_Labels", data=filtered_data, palette=cluster_colors, order=category_order)

# Customize x-axis labels
ax.set_xticklabels([category_labels.get(label, label) for label in category_order])

# Add title and labels
plt.title("Count of Cells for Each Cluster")
plt.xlabel("Categories")
plt.ylabel("Count of Cells")
plt.style.use('dark_background')

"""
# Add legend
ax.legend(title='Clusters', loc='upper right', labels=[f'Cluster {i}' for i in range(len(cluster_colors))])
"""

plt.savefig(Plot_path + f"Histogram_Clusters_{n_neighbors}.png", dpi=800, bbox_inches="tight")

# Show the plot
plt.show()






######################################################################
######################################################################
######################################################################
######################################################################


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

# Assuming 'data' is your DataFrame

# Filter out rows with Cluster_Labels equal to -1
filtered_data = data[data['Cluster_Labels'] != -1]

# Perform Kruskal-Wallis test
kruskal_result = kruskal(*[group['Cluster_Labels'] for name, group in filtered_data.groupby('categories')])

# Print Kruskal-Wallis result
print("Kruskal-Wallis Results:")
print(kruskal_result)


print("...")
print("...")
print("...")

# Perform Dunn's test for pairwise comparisons
dunn_result = posthoc_dunn(filtered_data, val_col='Cluster_Labels', group_col='categories')

# Print Dunn's test results
print("\nDunn's Test Results:")
print(dunn_result)







