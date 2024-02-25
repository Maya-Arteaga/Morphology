#https://www.datanovia.com/en/lessons/heatmap-in-r-static-and-interactive-visualization/
#https://r-charts.com/correlation/pheatmap/
#https://www.rdocumentation.org/packages/pheatmap/versions/1.0.12/topics/pheatmap


#install.packages("pheatmap")
library("pheatmap")

# Set the working directory
setwd("/Users/juanpablomayaarteaga/Desktop/Hilus/Prepro/")

# Define output directory for corr plots
Plot_path <- file.path("Output_images", "Plots", "Hierarchical_Distances")

# Create directory if it doesn't exist
if (!file.exists(Plot_path)) {
  dir.create(Plot_path, recursive = TRUE)
}

# Load the data from the CSV file
data <- read.csv("Output_images/Merged_Data/Morphology_UMAP_HDBSCAN_10.csv")


#set variables to asses
variables <- c("Area_soma", "Perimeter_soma", "Circularity_soma", "soma_compactness", "soma_orientation",
               "soma_feret_diameter", "soma_eccentricity", "soma_aspect_ratio", "End_Points", "Junctions",
               "Branches", "Initial_Points", "Total_Branches_Length", "ratio_branches", "polygon_area",
               "polygon_perimeters", "polygon_compactness", "polygon_eccentricities", "polygon_feret_diameters",
               "polygon_orientations", "cell_area", "cell_perimeter", "cell_circularity", "cell_compactness",
               "cell_orientation", "cell_feret_diameter", "cell_eccentricity", "cell_aspect_ratio",
               "sholl_max_distance", "sholl_crossing_processes", "sholl_num_circles", "cell_solidity",
               "cell_convexity", "UMAP_1", "UMAP_2")



df <- scale(data[variables])

# Convert 'Clusters_Labels' column to factor if necessary
data$Cluster_Labels <- as.factor(data$Cluster_Labels)


# Set names for the clusters
names(data$Cluster_Labels) <- paste("Cluster", levels(data$Cluster_Labels))


# Calculate pairwise distances between clusters
cluster_distances <- dist(data$Cluster_Labels)

# Convert the distance object to a square matrix
cluster_dist_matrix <- as.matrix(cluster_distances)


# Plot the heatmap with unique data and annotation
pheatmap(cluster_dist_matrix,
         cluster_rows = TRUE,          
         cluster_cols = TRUE,          
         cutree_rows = 7,         
         cutree_cols = 7,              
         fontsize_row = 10,            
         fontsize_col = 10,            
         show_rownames = FALSE,        
         show_colnames = FALSE,  
         legend=TRUE,
         main = "Clusters Distance Heatmap",  
         filename = file.path(Plot_path, "clusters_distance_heatmap.png")
)



