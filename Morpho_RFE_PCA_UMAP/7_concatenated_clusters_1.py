import cv2
import os
import numpy as np
import pandas as pd
from morpho import set_path, gammaCorrection
from collections import defaultdict

n_neighbors = 10

i_path = "/Users/juanpablomayaarteaga/Desktop/Hilus/Prepro/"
o_path = os.path.join(i_path, "Output_images/")
ID_path = set_path(o_path + f"ID_Just_clusters_{n_neighbors}/")

subject = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10"]
group = ["CNEURO1", "VEH", "CNEURO-01"]
treatment = ["ESC", "SS"]
tissue = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10"]

# Iterate over clusters and create composite images
for s in subject:
    for g in group:
        for tr in treatment:
            for ti in tissue:
                original_img = f"{s}_{g}_{tr}_{ti}_CA1_FOTO1_PROCESSED.tif"
                individual_img_path = os.path.join(o_path, f"{s}_{g}_{tr}_{ti}_CA1_FOTO1_PROCESSED/")
                csv_path = os.path.join(individual_img_path, "Data/")
                csv_file = f"{s}_{g}_{tr}_{ti}_Morphology_PCA_UMAP_HDBSCAN_{n_neighbors}.csv"

                if os.path.isfile(os.path.join(i_path, original_img)):
                    original_image = cv2.imread(os.path.join(i_path, original_img))
                    if original_image is not None:
                        original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2RGBA)
                        height, width, _ = original_image.shape

                        if os.path.isfile(os.path.join(csv_path, csv_file)):
                            data = pd.read_csv(os.path.join(csv_path, csv_file))
                            Cells_path = os.path.join(individual_img_path + "Cells/")
                            color_path = set_path(Cells_path + "Color_Cells/")

                            # Dictionary to store images for each cluster
                            cluster_images = defaultdict(list)

                            # Find the maximum height among all images
                            max_height = 0

                            # Loop through the rows of the 'data' DataFrame
                            for _, row in data.iterrows():
                                cell_number = row['Cell']
                                cluster = row['Cluster_Labels']
                                coordinates = eval(row['cell_positions'])
                                x1, y1, x2, y2 = coordinates

                                # Construct the image filename based on cell_number
                                image_filename = f"{cell_number}.tif"

                                # Check if the image file exists in the directory
                                if image_filename in os.listdir(color_path):
                                    input_image = os.path.join(color_path, image_filename)

                                    # Check if the input image file exists
                                    if os.path.isfile(input_image):
                                        # Read the colored image
                                        colored_image = cv2.imread(input_image, cv2.IMREAD_UNCHANGED)
                                        colored_image = colored_image[:, :, :3]
                                        rgba_image = cv2.cvtColor(colored_image, cv2.COLOR_BGR2BGRA)

                                        # Update the maximum height
                                        max_height = max(max_height, rgba_image.shape[0])

                                        # Append the image to the list for the corresponding cluster
                                        cluster_images[cluster].append(rgba_image[:, :, :4])

                            # Ensure empty lists for clusters that don't appear in the current row
                            for cluster_label in set(data['Cluster_Labels']):
                                if cluster_label not in cluster_images:
                                    cluster_images[cluster_label] = []

                                # Pad images to match the maximum height
                                cluster_images_padded = {cluster_label: [np.pad(image, ((0, max_height - image.shape[0]), (0, 0), (0, 0)), mode='constant', constant_values=0) for image in images] for cluster_label, images in cluster_images.items()}
    
                            # Save each cluster's composite image
                            # Iterate over clusters
                            for cluster_label, images in cluster_images_padded.items():
                                if images:
                                    horizontal_concatenation = np.concatenate(images, axis=1)
                            
                                    # Save the resulting image to a file without cropping
                                    resulting_image_path = os.path.join(ID_path, f"{s}_{g}_{tr}_{ti}_Cluster_{cluster_label}_UMAP_Concatenated.tif")
                                    cv2.imwrite(resulting_image_path, horizontal_concatenation)
