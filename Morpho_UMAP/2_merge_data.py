#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 21:44:50 2023

@author: juanpablomayaarteaga
"""


from morpho import set_path
import pandas as pd
import os





i_path = "/Users/juanpablomayaarteaga/Desktop/Hilus/Prepro/"
o_path = set_path(i_path + "/Output_images/")

csv_path= set_path(o_path + "/Merged_Data/")
Plot_path= set_path(o_path+"Plots/")




subject = ["R1", "R2"]
group = ["CNEURO1", "VEH"]
treatment = ["ESC", "SS"]
tissue = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"]

# Create an empty DataFrame to store the merged data
merged_data = pd.DataFrame()

for s in subject:
    for g in group:
        for tr in treatment:
            for ti in tissue:
                data_directory = f"{s}_{g}_{tr}_{ti}_CA1_FOTO1_PROCESSED/Data/"
                data_file = "Cell_Morphology.csv"
                data_path = os.path.join(o_path, data_directory, data_file)
                
                if os.path.isfile(data_path):
                    # Read each CSV file and merge them into a single DataFrame
                    df = pd.read_csv(data_path)
                    merged_data = pd.concat([merged_data, df])

# Save the merged DataFrame to a new CSV file

merged_data['Cell_ID'] = (merged_data['Cell'].astype(str) + "_" +
                   merged_data['group'] + "_" + 
                   merged_data['treatment']+ "_" +
                   merged_data['tissue'])


merged_data['categories'] = (merged_data['group'] + "_" +
                   merged_data['treatment'])

merged_data = merged_data[ ['Cell_ID'] + [col for col in merged_data.columns if col != 'Cell_ID'] ]
merged_data = merged_data[ ['categories'] + [col for col in merged_data.columns if col != 'categories'] ]



from sklearn.preprocessing import MinMaxScaler


"""
############# SCALING YOUR DATA FROM 0-1 
###### FILTERING "CELL" column 

# Select only the columns you want to scale (excluding "Cell" column)
numeric_columns = merged_data.select_dtypes(include=['float64', 'int64']).columns

# Exclude the "Cell" column from the selected columns
numeric_columns = numeric_columns[numeric_columns != 'Cell']

# Initialize the scaler
scaler = MinMaxScaler()

# Fit and transform the selected columns
merged_data[numeric_columns] = scaler.fit_transform(merged_data[numeric_columns])
"""



merged_csv_path = os.path.join(csv_path, "Morphology.csv")
merged_data.to_csv(merged_csv_path, index=False)
print(merged_csv_path)