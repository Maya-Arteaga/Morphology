#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 21:44:50 2023

@author: juanpablomayaarteaga
"""


from morpho import set_path
import pandas as pd
import os

#PATHS

i_path = "/Users/juanpablomayaarteaga/Desktop/Hilus/Prepro/"
o_path = set_path(i_path + "/Output_images/")
csv_path= set_path(o_path + "/Merged_Data/")
Plot_path= set_path(o_path+"Plots/")



#VARIABLES

subject = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10"]
group = ["CNEURO1", "VEH", "CNEURO-01"]
treatment = ["ESC", "SS"]
tissue = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10"]



# Empty DF

merged_data = pd.DataFrame()


# Loop to concatenate all the individual DF of  the samples to create one DF 
# which contains the information of all of them 

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



# Creating new columns to identify the file origin of the cells and its category

merged_data['Cell_ID'] = ("C"+ merged_data['Cell'].astype(str) + "_" +
                   merged_data['group'] + "_" + 
                   merged_data['treatment']+ "_" +
                   merged_data['tissue']+ "_" + 
                   merged_data['subject'])


merged_data['categories'] = (merged_data['group'] + "_" +
                   merged_data['treatment'])

merged_data = merged_data[ ['Cell_ID'] + [col for col in merged_data.columns if col != 'Cell_ID'] ]
merged_data = merged_data[ ['categories'] + [col for col in merged_data.columns if col != 'categories'] ]


# Save to a CSV file

merged_csv_path = os.path.join(csv_path, "Morphology.csv")
merged_data.to_csv(merged_csv_path, index=False)
print(merged_csv_path)


