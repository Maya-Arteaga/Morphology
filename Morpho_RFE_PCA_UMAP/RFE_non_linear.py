#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 14:20:07 2023

@author: juanpablomayaarteaga
"""

import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from morpho import set_path
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Record start time
start_time = time.time()

# Load the data
i_path = "/Users/juanpablomayaarteaga/Desktop/Hilus/Prepro/"
o_path = set_path(i_path + "/Output_images/")
csv_path = set_path(o_path + "/Merged_Data/")
Plot_path = set_path(o_path + "Plots/RFE/")

data = pd.read_csv(csv_path + "Morphology.csv")

# Specify the attributes you want to use in the decision tree
selected_attributes = [
    "Area_soma", "Perimeter_soma", "Circularity_soma", "soma_compactness",
    "soma_feret_diameter", "soma_eccentricity", "soma_aspect_ratio",
    "End_Points", "Junctions", "Branches", "Initial_Points",
    "Total_Branches_Length", "ratio_branches",
    "polygon_area", "polygon_perimeters", "polygon_compactness", "polygon_eccentricities",
    "polygon_feret_diameters",
    "cell_area", "cell_perimeter", "cell_circularity", "cell_compactness", "cell_feret_diameter",
    "cell_eccentricity", "cell_aspect_ratio", "cell_solidity", "cell_convexity",
    "sholl_max_distance", "sholl_crossing_processes", "sholl_num_circles"
]

# Specify the target variable
y = data['categories']

# Extract the selected features from the data
X = data[selected_attributes]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=24)

# Define the parameter grid for SVM hyperparameter tuning
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.01, 0.1, 1, 10]}

# Use GridSearchCV to find the best hyperparameters for non-linear SVM
nonlinear_svc = SVC(kernel="rbf", decision_function_shape='ovo')  # Use a non-linear kernel, e.g., RBF
grid_search_nonlinear = GridSearchCV(nonlinear_svc, param_grid, cv=5)
grid_search_nonlinear.fit(X_train, y_train)
best_nonlinear_svc = grid_search_nonlinear.best_estimator_

# Print the selected features
print("Selected Features:", selected_attributes)

# Transform the data to keep only the selected features
X_train_selected = X_train[selected_attributes]
X_test_selected = X_test[selected_attributes]

# Train a classifier on the selected features
best_nonlinear_svc.fit(X_train_selected, y_train)

# Make predictions on the test set
y_pred = best_nonlinear_svc.predict(X_test_selected)

# Evaluate the performance of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on Test Set:", accuracy)

# Additional metrics for multi-class classification
print("Classification Report:\n", classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Visualize the confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=best_nonlinear_svc.classes_, yticklabels=best_nonlinear_svc.classes_)
plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.savefig(Plot_path + "Confusion_Matrix_SFE_Nonlinear.png", dpi=800, bbox_inches="tight")
plt.tight_layout()

plt.show()

# Record end time
end_time = time.time()

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time} seconds")
