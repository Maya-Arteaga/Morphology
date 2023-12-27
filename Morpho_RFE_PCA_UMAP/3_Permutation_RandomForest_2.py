from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from morpho import set_path
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=24)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2', 0.5, None],  # Additional parameter
    'bootstrap': [True, False],  # Additional parameter
    'criterion': ['gini', 'entropy']  # Additional parameter
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get the best model from the grid search
best_rf_model = grid_search.best_estimator_

# Perform permutation feature importance with the best model
perm_importance = permutation_importance(best_rf_model, X_test, y_test, n_repeats=30, random_state=42)

# Get feature importances
feature_importances = perm_importance.importances_mean

# Print or use feature_importances as needed
print("Permutation Feature Importances:", feature_importances)

# Select features based on importance threshold (adjust as needed)
selected_features = [feature for feature, importance in zip(selected_attributes, feature_importances) if importance > 0]
print("Selected Features:", selected_features)

# Transform the data to keep only the selected features
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Cross-validation
cross_val_scores = cross_val_score(best_rf_model, X_train_selected, y_train, cv=5)
print("Cross-Validation Scores:", cross_val_scores)

# Train a classifier on the selected features
best_rf_model.fit(X_train_selected, y_train)

# Make predictions on the test set
y_pred_selected = best_rf_model.predict(X_test_selected)

# Evaluate the performance of the model
accuracy_selected = accuracy_score(y_test, y_pred_selected)
print("Accuracy on Test Set (Selected Features):", accuracy_selected)

# Additional metrics for multi-class classification
print("Classification Report (Selected Features):\n", classification_report(y_test, y_pred_selected))

# Compute the confusion matrix for selected features
cm_selected = confusion_matrix(y_test, y_pred_selected)

# Visualize the confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm_selected, annot=True, fmt='d', cmap='Blues', xticklabels=best_rf_model.classes_, yticklabels=best_rf_model.classes_)
plt.title('Confusion Matrix (Selected Features)', fontsize=16, fontweight='bold')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.tight_layout()
plt.show()

# Record end time
end_time = time.time()
print("Execution Time: {:.2f} seconds".format(end_time - start_time))


"""
Permutation Feature Importances: [ 0.0038808   0.0008316  -0.00949411 -0.0004158  -0.00734581 -0.00817741
 -0.0025641  -0.0014553   0.00880111  0.0029799  -0.00623701  0.
 -0.0001386  -0.01060291 -0.0011781  -0.00810811 -0.0004851  -0.00686071
 -0.00637561 -0.00540541 -0.00512821 -0.00533611 -0.00783091 -0.0043659
 -0.00810811 -0.0024255  -0.0015939   0.0036036   0.0022176  -0.0013167 ]
Selected Features: ['Area_soma', 'Perimeter_soma', 'Junctions', 'Branches', 'sholl_max_distance', 'sholl_crossing_processes']
Cross-Validation Scores: [0.33035714 0.35714286 0.28125    0.35714286 0.37053571]
Accuracy on Test Set (Selected Features): 0.3305613305613306
Classification Report (Selected Features):
                precision    recall  f1-score   support

CNEURO-01_ESC       0.37      0.55      0.44       113
  CNEURO1_ESC       0.43      0.31      0.36       105
   CNEURO1_SS       0.36      0.24      0.29        99
      VEH_ESC       0.23      0.23      0.23        81
       VEH_SS       0.24      0.25      0.25        83

     accuracy                           0.33       481
    macro avg       0.33      0.32      0.31       481
 weighted avg       0.34      0.33      0.32       481
 """
