#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 17:52:57 2023

@author: juanpablomayaarteaga
"""


from morpho import set_path
import pandas as pd

# Load the data from the CSV file
i_path = "/Users/juanpablomayaarteaga/Desktop/Hilus/Prepro/"
o_path = set_path(i_path + "/Output_images/")
csv_path = set_path(o_path + "Merged_Data/")
store_path = set_path(o_path + "Merged_Data/Descriptive_Statistic/")
Plot_path = set_path(o_path + "Plots/Frequencies/")

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

# Use the correct column name for categories
categories_column = "categories"



##################################################################################################################
##################################################################################################################
##################################### DESCRIPTIVE STATISTICS  #################################################################
##################################################################################################################
##################################################################################################################




# Create an empty DataFrame to store the results
result_df = pd.DataFrame(index=variables)

# Iterate over each variable and calculate the desired statistics for each category
for variable in variables:
    for category in categories:
        subset_data = data[data[categories_column] == category][variable]
        result_df.loc[variable, f"{category} Mean"] = subset_data.mean()
        result_df.loc[variable, f"{category} Median"] = subset_data.median()
        result_df.loc[variable, f"{category} Variance"] = subset_data.var()
        result_df.loc[variable, f"{category} Std Dev"] = subset_data.std()
        result_df.loc[variable, f"{category} 25th Percentile"] = subset_data.quantile(0.25)
        result_df.loc[variable, f"{category} 50th Percentile"] = subset_data.quantile(0.50)
        result_df.loc[variable, f"{category} 75th Percentile"] = subset_data.quantile(0.75)



# Display the result DataFrame
print(result_df)
result_df = result_df.round(2)
# Save the results to a CSV file
result_df.to_csv(store_path + "Variables_Descriptive_Statistics_Parameters.csv", index=True)



##################################################################################################################
##################################################################################################################
######################################### SUBSET STATISTICS  #################################################################
##################################################################################################################
##################################################################################################################




# Create separate DataFrames for each statistic
mean_df = result_df.filter(like="Mean")
median_df = result_df.filter(like="Median")
variance_df = result_df.filter(like="Variance")
std_dev_df = result_df.filter(like="Std Dev")
percentile_25_df = result_df.filter(like="25th Percentile")
percentile_50_df = result_df.filter(like="50th Percentile")
percentile_75_df = result_df.filter(like="75th Percentile")

# Save each DataFrame to a CSV file
mean_df.to_csv(store_path + "Mean.csv", index=True)
median_df.to_csv(store_path + "Median.csv", index=True)
variance_df.to_csv(store_path + "Variance.csv", index=True)
std_dev_df.to_csv(store_path + "Std_Dev.csv", index=True)
percentile_25_df.to_csv(store_path + "Percentile_25.csv", index=True)
percentile_50_df.to_csv(store_path + "Percentile_50.csv", index=True)
percentile_75_df.to_csv(store_path + "Percentile_75.csv", index=True)





##################################################################################################################
##################################################################################################################
######################################### CELL COUNTING  #################################################################
##################################################################################################################
##################################################################################################################

# Count the occurrences of each category in the original data
category_counts = data[categories_column].value_counts()

# Rename the index name to 'Cell Number'
category_counts.index.name = 'Cell Number'

# Save the category counts to a CSV file with 'Cell Number' as the column name
category_counts.to_csv(store_path + "Cell_Counting.csv", header=None)

# Read the CSV file and save it with the correct header order
category_counts_df = pd.read_csv(store_path + "Cell_Counting.csv", names=["Categories", "Cell Number"])

# Save the corrected DataFrame to the CSV file
category_counts_df.to_csv(store_path + "Cell_Counting.csv", index=False)

# Display the corrected category counts DataFrame
print(category_counts_df)




import pandas as pd

# Assuming you have the original DataFrame as category_counts_df

# Define the category order
category_order = ["VEH_SS", "VEH_ESC", "CNEURO-01_ESC", "CNEURO1_ESC", "CNEURO1_SS"]

# Create a Categorical data type with the desired order
category_dtype = pd.CategoricalDtype(categories=category_order, ordered=True)

# Change the 'Categories' column to the new Categorical data type
category_counts_df["Categories"] = category_counts_df["Categories"].astype(category_dtype)

# Sort the DataFrame based on the new order
category_counts_df = category_counts_df.sort_values(by="Categories")

# Save the corrected DataFrame to the CSV file
category_counts_df.to_csv(store_path + "Cell_Counting_reorder.csv", index=False)

# Display the corrected category counts DataFrame
print(category_counts_df)






# Select relevant columns
selected_columns = ["categories", "Cluster_Labels"]

# Filter data for the selected categories
filtered_data = data[data["categories"].isin(categories)]

# Create a DataFrame to count the frequency of each cluster within each category
cluster_counts = filtered_data.groupby(["categories", "Cluster_Labels"]).size().unstack(fill_value=0)

# Add an extra row and column for the total counts
cluster_counts["Total"] = cluster_counts.sum(axis=1)
cluster_counts.loc["Total"] = cluster_counts.sum()

# Display the result
print(cluster_counts)

# Save the result to a CSV file
cluster_counts.to_csv(store_path + "Cluster_Frequency_By_Category.csv", index=True)










import pandas as pd

# Assuming you have the original DataFrame as cluster_counts

# Define the category order
category_order = ["VEH_SS", "VEH_ESC", "CNEURO-01_ESC", "CNEURO1_ESC", "CNEURO1_SS"]

# Select relevant columns
selected_columns = ["categories", "Cluster_Labels"]

# Filter data for the selected categories
filtered_data = data[data["categories"].isin(categories)]

# Reorder the "categories" column
filtered_data["categories"] = pd.Categorical(filtered_data["categories"], categories=category_order, ordered=True)

# Create a DataFrame to count the frequency of each cluster within each category
cluster_counts = filtered_data.groupby(["categories", "Cluster_Labels"]).size().unstack(fill_value=0)

# Add an extra row and column for the total counts
cluster_counts["Total"] = cluster_counts.sum(axis=1)
cluster_counts.loc["Total"] = cluster_counts.sum()

# Display the result
print(cluster_counts)

# Save the result to a CSV file
cluster_counts.to_csv(store_path + "Cluster_Frequency_By_Category_reorder.csv", index=True)









import matplotlib.pyplot as plt

# Custom color palette
cluster_colors = {
    0: "orangered",
    1: "crimson",
    4: "paleturquoise",
    2: "gold",
    3: "limegreen",
    5: "mediumturquoise",
}

# Drop the "Total" row and column for plotting
cluster_counts_plot = cluster_counts.drop(index="Total", columns="Total")

# Map original category labels to custom labels
category_labels = {"VEH_SS": "VEH SS", "CNEURO1_ESC": "CNEURO 1.0 ESC", "VEH_ESC": "ESC", "CNEURO-01_ESC": "CNEURO 0.1 ESC", "CNEURO1_SS": "CNEURO 1.0 SS"}
custom_category_labels = cluster_counts_plot.index.map(category_labels)

# Plot the cluster frequencies with custom colors
ax = cluster_counts_plot.plot(kind="bar", stacked=True, figsize=(10, 6), color=[cluster_colors[col] for col in cluster_counts_plot.columns])

# Customize x-axis labels
ax.set_xticklabels(custom_category_labels)

plt.title("")
plt.xlabel("Categories")
plt.ylabel("Cluster Frequencies")
plt.legend(title="Cluster Labels", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.savefig(Plot_path + "Stack_frequencies_Clusters.png", dpi=800, bbox_inches="tight")
plt.tight_layout()
# Show the plot
plt.show()



import matplotlib.pyplot as plt

# Custom color palette
cluster_colors = {
    0: "orangered",
    1: "crimson",
    4: "paleturquoise",
    2: "gold",
    3: "limegreen",
    5: "mediumturquoise",
}

# Drop the "Total" row and column for plotting
cluster_counts_plot = cluster_counts.drop(index="Total", columns="Total")

# Map original category labels to custom labels
category_labels = {"VEH_SS": "VEH SS", "CNEURO1_ESC": "CNEURO 1.0", "VEH_ESC": "ESC", "CNEURO-01_ESC": "CNEURO 0.1", "CNEURO1_SS": "CNEURO 1.0 SS"}
custom_category_labels = cluster_counts_plot.index.map(category_labels)

# Normalize by category (convert frequencies to percentages)
cluster_percentages = cluster_counts_plot.div(cluster_counts_plot.sum(axis=1), axis=0) * 100

# Plot the cluster frequencies as percentages with custom colors
ax = cluster_percentages.plot(kind="bar", stacked=True, figsize=(10, 6), color=[cluster_colors[col] for col in cluster_percentages.columns])

# Customize x-axis labels
ax.set_xticklabels(custom_category_labels)

plt.title("")
plt.xlabel("Categories")
plt.ylabel("Cluster Percentages")
plt.legend(title="Cluster Labels", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.savefig(Plot_path + "Stack_percentages_Clusters.png", dpi=800, bbox_inches="tight")
plt.tight_layout()
# Show the plot
plt.show()








