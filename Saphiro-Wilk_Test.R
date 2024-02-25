# Load necessary libraries
library(rstatix)

# Set the working directory
setwd("/Users/juanpablomayaarteaga/Desktop/Hilus/Prepro/")

# Load the data from the CSV file
data <- read.csv("Output_images/Merged_Data/Morphology_UMAP_HDBSCAN_10.csv")

# List of variables to create distribution plots for
variables <- c("Area_soma", "Perimeter_soma", "Circularity_soma", "soma_compactness", "soma_orientation",
               "soma_feret_diameter", "soma_eccentricity", "soma_aspect_ratio", "End_Points", "Junctions",
               "Branches", "Initial_Points", "Total_Branches_Length", "ratio_branches", "polygon_area",
               "polygon_perimeters", "polygon_compactness", "polygon_eccentricities", "polygon_feret_diameters",
               "polygon_orientations", "cell_area", "cell_perimeter", "cell_circularity", "cell_compactness",
               "cell_orientation", "cell_feret_diameter", "cell_eccentricity", "cell_aspect_ratio",
               "sholl_max_distance", "sholl_crossing_processes", "sholl_num_circles", "cell_solidity",
               "cell_convexity", "UMAP_1", "UMAP_2")

# Perform Shapiro-Wilk test for normality for each variable
for (variable in variables) {
  test_result <- shapiro_test(data[, variable])
  print(paste("Variable:", variable))
  print(test_result)
}
