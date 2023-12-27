#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 19:56:19 2023

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

# Display the correlation matrix heatmap
selected_data = data[selected_attributes]
correlation_matrix = selected_data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', annot_kws={"size":8})
plt.title('Correlation Matrix Heatmap', fontsize=16, fontweight='bold')
plt.savefig(Plot_path + "Correlation_Matrix_ALL.png", dpi=800)
plt.tight_layout()
plt.show()

# Specify the target variable
y = data['categories']

# Extract the selected features from the data
X = data[selected_attributes]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=24)

# Create an SVM classifier for multi-class classification
svc = SVC(kernel="linear", decision_function_shape='ovr')  # 'ovr' stands for One-vs-Rest

# Create the RFE model and select the number of features to retain
num_features_to_keep = X.shape[1] // 2
rfe = RFE(estimator=svc, n_features_to_select=num_features_to_keep)

# Fit the RFE model on the training data
rfe.fit(X_train, y_train)

# Get the ranking of each feature (1 means selected, 0 means not selected)
feature_ranking = rfe.ranking_

# Print the selected features
selected_features = [feature for feature, rank in zip(selected_attributes, feature_ranking) if rank == 1]
print("Selected Features:", selected_features)

# Transform the data to keep only the selected features
X_train_selected = rfe.transform(X_train)
X_test_selected = rfe.transform(X_test)

# Hyperparameter tuning using GridSearchCV
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.01, 0.1, 1, 10]}
grid_search = GridSearchCV(SVC(kernel="rbf", decision_function_shape='ovr'), param_grid, cv=5)
grid_search.fit(X_train_selected, y_train)
best_svc = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_svc.predict(X_test_selected)

# Evaluate the performance of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on Test Set:", accuracy)

# Additional metrics for multi-class classification
print("Classification Report:\n", classification_report(y_test, y_pred))

# Display the confusion matrix
from sklearn.metrics import confusion_matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=best_svc.classes_, yticklabels=best_svc.classes_)
plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig(Plot_path + "Confusion_Matrix_SFE.png", dpi=800)
plt.tight_layout()
plt.show()

# Record end time
end_time = time.time()

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time} seconds")
