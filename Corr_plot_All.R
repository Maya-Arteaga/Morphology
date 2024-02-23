# Load necessary libraries
library(corrplot)


# Set the working directory
setwd("/Users/juanpablomayaarteaga/Desktop/Hilus/Prepro/")

# Define output directory for corr plots
Plot_path <- file.path("Output_images", "Plots", "Corr")

# Create directory if it doesn't exist
if (!file.exists(Plot_path)) {
  dir.create(Plot_path, recursive = TRUE)
}

# Set the filename for the plot
plot_filename <- file.path(Plot_path, "corr_plot_All.png")

# Load the data from the CSV file
data <- read.csv("Output_images/Merged_Data/Morphology_UMAP_HDBSCAN_10.csv")

# Define the desired DPI (e.g., 300)
dpi <- 300
# Calculate the width and height of the image based on the desired DPI
width_inches <- 10  # Adjust as needed
height_inches <- 10  # Adjust as needed

# Calculate the width and height in pixels
width_pixels <- dpi * width_inches
height_pixels <- dpi * height_inches

# Open a PNG graphics device with the desired width, height, and DPI
png(plot_filename, width = width_pixels, height = height_pixels, res = dpi)


variables <- c("Area_soma", "Perimeter_soma", "Circularity_soma", "soma_compactness", "soma_orientation",
               "soma_feret_diameter", "soma_eccentricity", "soma_aspect_ratio", "End_Points", "Junctions",
               "Branches", "Initial_Points", "Total_Branches_Length", "ratio_branches", "polygon_area",
               "polygon_perimeters", "polygon_compactness", "polygon_eccentricities", "polygon_feret_diameters",
               "polygon_orientations", "cell_area", "cell_perimeter", "cell_circularity", "cell_compactness",
               "cell_orientation", "cell_feret_diameter", "cell_eccentricity", "cell_aspect_ratio",
               "sholl_max_distance", "sholl_crossing_processes", "sholl_num_circles", "cell_solidity",
               "cell_convexity", "UMAP_1", "UMAP_2")




#@corrplot(cor(data[variables]))

corrplot.mixed(
  cor(data[variables]),
  upper="square",
  lower="number",
  addgrid.col="black",
  tl.col="black",
  tl.cex = 0.2, #Font size,
  number.cex=0.6
)


# Close the graphics device
dev.off()


