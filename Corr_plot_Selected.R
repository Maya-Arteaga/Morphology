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
plot_filename <- file.path(Plot_path, "corr_plot_Selected9.png")

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




# Adjust the transparency of the default color palette
adjusted_palette <- adjustcolor(default_palette, alpha.f = 1.2)

# Open a PNG graphics device with the desired width, height, and DPI
png(plot_filename, width = width_pixels, height = height_pixels, res = dpi)




features_to_assess <- c('Circularity_soma', 'soma_compactness', 'soma_eccentricity', 
                        'soma_aspect_ratio', 'Junctions', 'Initial_Points', 
                        'ratio_branches', 'polygon_eccentricities', 'cell_compactness', 
                        'cell_feret_diameter', 'cell_eccentricity', 'cell_aspect_ratio', 
                        'cell_solidity', 'cell_convexity', 'sholl_num_circles')


# corrplot.mixed(
#   cor(data[features_to_assess]),
#   upper="number",
#   lower="circle",
#   addgrid.col="black",
#   tl.col="black",
#   tl.cex = 0.5, #Font size,
#   tl.srt=45,
#   number.cex=0.9,
#   font = 2,
#   number.font=2
# )

corrplot(
  cor(data[features_to_assess]),
  addCoef.col = 'black',
  type = "upper",
  method = "circle",  # Specify the method parameter here
  number.cex=0.9,
  font = 2,
  tl.pos = 'd',
  tl.cex = 0.4,
  tl.col="darkred",
  
)


# Close the graphics device
dev.off()


