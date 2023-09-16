




import seaborn as sns
from morpho import set_path
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import pandas as pd
import os




i_path="/Users/juanpablomayaarteaga/Desktop/Prueba_Morfo/"

o_path= set_path(i_path+"/Output_images/")

csv_path= set_path(o_path+"Data/")


Plot_path= set_path(o_path+"Plots/")


# Load the data from the CSV file
data = pd.read_csv(csv_path + "Cell_Morphology.csv")



# Define the columns to assess and columns to erase
######### 2D 
cell_soma_branches = ["Area_soma", "Perimeter_soma","soma_eccentricity", "soma_aspect_ratio", "Circularity_soma", "soma_compactness", "soma_feret_diameter", "Branches", "Junctions", "End_Points", "Initial_Points"]
cell_geometry = ["polygon_compactness", "eccentricities", "feret_diameters"]

######### 3D
cell_soma = ["Area_soma", "Perimeter_soma", "Circularity_soma", "soma_compactness", "soma_feret_diameter", "soma_eccentricity", "soma_aspect_ratio"]
cell_branches = ["Branches", "Junctions", "End_Points", "Initial_Points"]


#columns_3d = cell_branches + cell_branches + cell_geometry

columns_to_assess = cell_soma_branches + cell_geometry 

columns_to_erase= columns_to_assess

# Scale the selected columns
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[columns_to_assess])

#scaled_3d_data = scaler.fit_transform(data[columns_3d])


# Create a DataFrame with scaled values and original column names
scaled_df = pd.DataFrame(scaled_data, columns=columns_to_assess,)


#scaled_3d_data = pd.DataFrame(scaled_3d_data, columns=columns_to_assess)


# Drop the columns specified in erase_columns
data = data.drop(columns_to_erase, axis=1)

# Concatenate the columns from scaled_df after the columns in the original DataFrame data
scaled_data = pd.concat([data, scaled_df], axis=1)




# Save the combined DataFrame to a new CSV file
scaled_data.to_csv(csv_path + "Cell_Morphology_Scaled.csv", index=False)






# Apply PCA for "soma and branches" category
pca_cell_soma_branches = PCA(n_components=1)
cell_soma_branches_dimension = pca_cell_soma_branches.fit_transform(scaled_data[cell_soma_branches])

# Apply PCA for "geometry" category
pca_cell_geometry = PCA(n_components=1)
cell_geometry_dimension = pca_cell_geometry.fit_transform(scaled_data[cell_geometry])


# Apply PCA for "soma" category
pca_cell_soma = PCA(n_components=1)
cell_soma_dimension = pca_cell_soma.fit_transform(scaled_data[cell_soma])


# Apply PCA for "branches" category
pca_cell_branches = PCA(n_components=1)
cell_branches_dimension = pca_cell_branches.fit_transform(scaled_data[cell_branches])


#Add cell_soma_branches_dimension and cell_geometry_dimension to the dataframe "scaled_data"
scaled_data["cell_soma_branches_dimension"] = cell_soma_branches_dimension
scaled_data["cell_geometry_dimension"] = cell_geometry_dimension

scaled_data["cell_soma_dimension"] = cell_soma_dimension
scaled_data["cell_branches_dimension"] = cell_branches_dimension


scaled_data.to_csv(csv_path + "Cell_Morphology_Scaled.csv", index=False)







################ K MEANS
# After making the PCA: asses the better k means by Elbow method 
# to the cell_soma_branches_dimension vs cell_geometry_dimension


SSD = []
K = range(1, 11)

# Define the columns for K-means clustering
columns_for_kmeans = ["cell_soma_dimension", "cell_geometry_dimension", "cell_branches_dimension"]

for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(scaled_data[columns_for_kmeans])  # Specify the columns here
    SSD.append(kmeanModel.inertia_)

plt.plot(K, SSD, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of squared distances')
plt.title('Elbow Method for Optimal k')

# Save the plot as "Elbow_plot_non-scaled.png" at csv_path
plot_filename = Plot_path + "Elbow_plot_Scaled.png"
plt.savefig(plot_filename)


plt.show()




# After making the PCA: asses the better k means by Silhouette Score method
# to the cell_soma_branches_dimension vs cell_geometry_dimension






from sklearn.metrics import silhouette_score

silhouette_scores = []
K = range(2, 11)

for k in K:
    kmeanModel = KMeans(n_clusters=k)
    labels = kmeanModel.fit_predict(scaled_data[columns_for_kmeans])
    silhouette_avg = silhouette_score(scaled_data[columns_for_kmeans], labels)
    silhouette_scores.append(silhouette_avg)

plt.plot(K, silhouette_scores, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Optimal k')

# Save the plot as "Elbow_plot_non-scaled.png" at csv_path
plot_filename = Plot_path + "Silhouette_plot_Scaled.png"
plt.savefig(plot_filename)

plt.show()












# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(scaled_data['cell_soma_branches_dimension'], scaled_data['cell_geometry_dimension'])

# Labeling the axes
plt.xlabel('Cell')
plt.ylabel('Geometry')

# Add labels to data points (Cell identifier)
for i, cell in enumerate(scaled_data['Cell']):
    plt.annotate(cell, (scaled_data['cell_soma_branches_dimension'][i], scaled_data['cell_geometry_dimension'][i]), fontsize=6, verticalalignment='center')

# Add a title
plt.title("PCA: Cell vs Geometry")

# Save the plot as an image file
plot_filename = os.path.join(Plot_path, 'Cell_vs_Geometry_Scaled.png')
plt.savefig(plot_filename, dpi=500, bbox_inches='tight')  # Adjust DPI value as needed
#plt.show()  # Display the scatter plot

















import matplotlib.pyplot as plt

# Assuming you have a DataFrame called scaled_data
# Extract the columns for the x, y, and z axes
x = scaled_data["cell_soma_dimension"]
y = scaled_data["cell_geometry_dimension"]
z = scaled_data["cell_branches_dimension"]

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with x, y, and z as axes
ax.scatter(x, y, z, c='b', marker='o')  # You can customize the color and marker style

# Label the axes
ax.set_xlabel("Soma Dimension")
ax.set_ylabel("Geometry Dimension")
ax.set_zlabel("Branches Dimension")




# Set the title
ax.set_title("PCA: Soma vs Branches vs Geometry")

# Show the plot
plt.show()







"""

# Apply K-Means clustering to the reduced dimensions


n_clusters = 7
kmeans = KMeans(n_clusters=n_clusters, random_state=42)

# Combine the reduced dimensions into one feature matrix
feature_matrix = pd.DataFrame({'Cell': scaled_data['cell_soma_branches_dimension'], 'Geometry': scaled_data['cell_geometry_dimension']})

# Fit the K-Means model to the feature matrix
kmeans.fit(feature_matrix)

# Add cluster labels to the DataFrame
scaled_data['Cluster'] = kmeans.labels_

# Create a scatter plot to visualize the clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x='cell_soma_branches_dimension', y='cell_geometry_dimension', 
                hue=scaled_data['Cluster'], 
                data=scaled_data,  # Use scaled_data for data source
                palette='viridis')

plt.xlabel('Cell')
plt.ylabel('Geometry')
plt.title("PCA: Cell vs Geometry (Scaled)")

# Add labels to data points (Cell identifier) using the 'Cell' column from scaled_data
for i, cell in enumerate(scaled_data['Cell']):
    plt.annotate(cell, (scaled_data['cell_soma_branches_dimension'][i], scaled_data['cell_geometry_dimension'][i]), fontsize=6, verticalalignment='center')

# Save the plot as an image file
clustered_plot_filename = os.path.join(Plot_path, f'K-means_cell_vs_geometry_Scaled_{n_clusters}.png')
plt.savefig(clustered_plot_filename, dpi=500, bbox_inches='tight')
plt.show()
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import os

# Define cluster colors with color names
cluster_colors = {
    0: 'cyan',
    1: 'limegreen',
    2: 'red',
    3: 'blue',
    4: 'magenta',
    5: 'yellow',
    6: 'purple',
    7: 'orange',
}

# Assuming you have a DataFrame called scaled_data
# Extract the columns for the x, y, and z axes
x = scaled_data["cell_soma_dimension"]
y = scaled_data["cell_geometry_dimension"]
z = scaled_data["cell_branches_dimension"]

# Number of clusters
n_clusters = 8

# Fit K-means to the data
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(scaled_data[["cell_soma_dimension", "cell_geometry_dimension", "cell_branches_dimension"]])

# Get cluster labels for each data point
cluster_labels = kmeans.labels_

# Map cluster labels to cluster colors
cluster_colors_list = [cluster_colors[label] for label in cluster_labels]

# Create a 3D scatter plot with cluster colors
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with x, y, and z as axes, colored by cluster
scatter = ax.scatter(x, y, z, c=cluster_colors_list, marker='o')

# Label the axes
ax.set_xlabel("Soma Dimension")
ax.set_ylabel("Geometry Dimension")
ax.set_zlabel("Branches Dimension", labelpad=0)
#ax.set_zlabel("")  # Clear the default label
#ax.text(x.max(), y.max(), z.min(), "Branches Dimension", fontsize=12, verticalalignment='top', horizontalalignment='left')


# Set the title
ax.set_title("Branches vs Soma vs Geometry")

# Add labels to data points (Cell identifier)
for i, cell in enumerate(scaled_data['Cell']):
    ax.text(x[i], y[i], z[i], cell, fontsize=6, verticalalignment='center')

# Save the plot as an image file
clustered_plot_filename = os.path.join(Plot_path, f'K-means_Scaled_{n_clusters}.png')
plt.savefig(clustered_plot_filename, dpi=500, bbox_inches='tight')

# Show the plot
plt.show()

# Print the path where the image is saved
print("Clustered plot saved in:", clustered_plot_filename)








import pandas as pd
from sklearn.cluster import KMeans

# Number of clusters
n_clusters = 8

# Fit K-means to the data
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
scaled_data["Cluster"] = kmeans.fit_predict(scaled_data[["cell_soma_dimension", "cell_geometry_dimension", "cell_branches_dimension"]])



# Save the DataFrame to a CSV file
output_filename = csv_path +f"Morphology_Scaled_{n_clusters}.csv"
scaled_data.to_csv(output_filename, index=False)

print(f"DataFrame saved as {output_filename}")




# Fit K-means to the data
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
scaled_data["Cluster"] = kmeans.fit_predict(scaled_data[["cell_soma_dimension", "cell_geometry_dimension", "cell_branches_dimension"]])



