# Load necessary libraries
library(ggstatsplot)
library(PMCMRplus)
library(ggplot2)

# Set the working directory
setwd("/Users/juanpablomayaarteaga/Desktop/Hilus/Prepro/")

# Define output directory for violin plots
Violin_path <- file.path("Output_images", "Plots", "Clusters", "R")

# Load the data from the CSV file
data <- read.csv("Output_images/Merged_Data/Morphology_UMAP_HDBSCAN_10.csv")

# Define category column
category_column <- "Cluster_Labels"

# Define hue order
hue_order <- c("0", "1", "2",  "3", "4", "5")

# Define colors
colors <- c("#FF0000", "#FF8C00", "#FFFF00", "#008000", "#48D1CC", "#AFEEEE")

# List of variables to create distribution plots for
variables <- c("Area_soma", "Perimeter_soma", "Circularity_soma", "soma_compactness", "soma_orientation",
               "soma_feret_diameter", "soma_eccentricity", "soma_aspect_ratio", "End_Points", "Junctions",
               "Branches", "Initial_Points", "Total_Branches_Length", "ratio_branches", "polygon_area",
               "polygon_perimeters", "polygon_compactness", "polygon_eccentricities", "polygon_feret_diameters",
               "polygon_orientations", "cell_area", "cell_perimeter", "cell_circularity", "cell_compactness",
               "cell_orientation", "cell_feret_diameter", "cell_eccentricity", "cell_aspect_ratio",
               "sholl_max_distance", "sholl_crossing_processes", "sholl_num_circles", "cell_solidity",
               "cell_convexity", "UMAP_1", "UMAP_2")

# Convert category_column to factor
data[[category_column]] <- factor(data[[category_column]])

# changing palette with custom values
for (variable in variables) {
  p <- ggbetweenstats(
    data = data,
    x = !!rlang::sym(category_column),
    y = !!rlang::sym(variable),
    type = "nonparametric",
    plot.type = "violin",
    pairwise.comparisons = TRUE,
    pairwise.display = "significant",
    adjust = "bonferroni",  # Apply Bonferroni correction
    centrality.plotting = FALSE,
    bf.message = FALSE
  ) +
    ggtitle(paste("Kruskal-Wallis Test for", variable)) +
    labs(y = variable) +
    scale_color_manual(values = colors) +  # Assign colors to clusters
    scale_x_discrete(limits = hue_order) +  # Set order
    theme(
      text = element_text(size=15, family = "times new roman"),
      axis.text.x = element_text(size=15),
      plot.title = element_text(hjust = 0.5)  # Center the title
    )

  # Save the plot
  ggsave(filename = paste0(Violin_path, "/", variable, "_violin_plot.png"), plot = p, width = 12, height = 8, dpi = 300)
}
