#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 13:02:07 2023

@author: juanpablomayaarteaga
"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from morpho import set_path
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import confusion_matrix

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=24)

# Train a RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Perform permutation feature importance
perm_importance = permutation_importance(rf_model, X_test, y_test, n_repeats=30, random_state=42)

# Get feature importances
feature_importances = perm_importance.importances_mean

# Print or use feature_importances as needed
print("Permutation Feature Importances:", feature_importances)

# Select features based on importance threshold (adjust as needed)
selected_features = [feature for feature, importance in zip(selected_attributes, feature_importances) if importance > 0.05]
print("Selected Features:", selected_features)

# Transform the data to keep only the selected features
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Train a classifier on the selected features
rf_model_selected = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_selected.fit(X_train_selected, y_train)

# Make predictions on the test set
y_pred_selected = rf_model_selected.predict(X_test_selected)

# Evaluate the performance of the model
accuracy_selected = accuracy_score(y_test, y_pred_selected)
print("Accuracy on Test Set (Selected Features):", accuracy_selected)

# Additional metrics for multi-class classification
print("Classification Report (Selected Features):\n", classification_report(y_test, y_pred_selected))

# Compute the confusion matrix for selected features
cm_selected = confusion_matrix(y_test, y_pred_selected)

# Visualize the confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm_selected, annot=True, fmt='d', cmap='Blues', xticklabels=rf_model_selected.classes_, yticklabels=rf_model_selected.classes_)
plt.title('Confusion Matrix (Selected Features)', fontsize=16, fontweight='bold')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.tight_layout()
plt.show()
