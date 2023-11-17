#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 16:48:17 2023

@author: juanpablomayaarteaga
"""
from morpho import set_path
import pandas as pd
import umap
import matplotlib.pyplot as plt
import hdbscan
import seaborn as sns
import numpy as np



# Load the data from the CSV file
i_path = "/Users/juanpablomayaarteaga/Desktop/Hilus/Prepro/"
o_path = set_path(i_path + "/Output_images/")
csv_path = set_path(o_path + "/Merged_Data/")
Plot_path = set_path(o_path + "Plots/")

data = pd.read_csv(csv_path + "Morphology.csv")



from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Specify the attributes you want to use in the decision tree
selected_attributes = [
    "Area_soma", "Perimeter_soma",	"Circularity_soma",	"soma_compactness",	
    "soma_feret_diameter",	"soma_eccentricity",	"soma_aspect_ratio",
	"End_Points",	"Junctions",	"Branches",	"Initial_Points",	
    "Total_Branches_Length","ratio_branches",
	"polygon_area",	"polygon_perimeters",	"polygon_compactness",	"polygon_eccentricities",
    "polygon_feret_diameters",	
    "cell_area",	"cell_perimeter",	"cell_circularity",	"cell_compactness",	"cell_feret_diameter",
	"cell_eccentricity",	"cell_aspect_ratio", "cell_solidity", "cell_convexity",
	"sholl_max_distance",	"sholl_crossing_processes",	"sholl_num_circles"
    
]

# Extract features and target variable using selected attributes
X = data[selected_attributes]  # Features
y = data['categories']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Display classification report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Display the decision tree rules
tree_rules = export_text(clf, feature_names=selected_attributes)
print("Decision Tree Rules:\n", tree_rules)


from sklearn.tree import plot_tree

plt.figure(figsize=(15, 10))
plot_tree(clf, feature_names=selected_attributes, class_names=clf.classes_, filled=True, rounded=True)
plt.show()



# Visualize feature importances
feature_importances = pd.Series(clf.feature_importances_, index=selected_attributes)
sns.barplot(x=feature_importances, y=feature_importances.index, orient='h')
plt.title('Feature Importances')
plt.show()










from sklearn.model_selection import GridSearchCV

# Define the hyperparameters and their potential values
param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Create a decision tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Retrain the model with the best hyperparameters
best_clf = grid_search.best_estimator_
best_clf.fit(X_train, y_train)





# Train the decision tree with the best hyperparameters
best_clf.fit(X_train, y_train)

# Access feature importances
feature_importances = best_clf.feature_importances_

# Create a DataFrame to display feature importances
importances_df = pd.DataFrame({'Feature': selected_attributes, 'Importance': feature_importances})
importances_df = importances_df.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importances_df, orient='h')
plt.title('Feature Importances')
plt.show()






# Assuming importances_df is the DataFrame containing feature importances
cumulative_importance = importances_df['Importance'].cumsum()

# Set a threshold (e.g., 95% cumulative importance)
threshold = 0.95

# Select features that exceed the threshold
selected_features = importances_df.loc[cumulative_importance <= threshold, 'Feature']

# Display selected features
print("Selected Features:")
print(selected_features)




