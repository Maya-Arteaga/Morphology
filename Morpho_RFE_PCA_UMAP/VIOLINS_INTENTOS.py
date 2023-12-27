import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from morpho import set_path
from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn
from matplotlib.colors import LinearSegmentedColormap

# Load the data from the CSV file
i_path = "/Users/juanpablomayaarteaga/Desktop/Hilus/Prepro/"
o_path = set_path(i_path + "/Output_images/")
csv_path = set_path(o_path + "/Merged_Data/")
Distributions_path = set_path(o_path + "Plots/Distributions/")
Violin_path = set_path(o_path + "Plots/Violins/")
Dunn_path = set_path(o_path + "Plots/Dunn/")

data = pd.read_csv(csv_path + "Morphology_PCA_UMAP_HDBSCAN_15.csv")

# Column for categories
category_column = "categories"
# variables = ["UMAP_1", "UMAP_2"]
variables = ["UMAP_1"]

# Define colors and labels for each category
colors = {'VEH_ESC': 'red', 'CNEURO1_ESC': 'blue', 'VEH_SS': 'green'}

# Specify the order of categories
hue_order = ['VEH_SS', 'CNEURO1_ESC', 'VEH_ESC']

# Define the desired order for Dunn results
dunn_order = [('VEH_SS', 'CNEURO1_ESC'), ('CNEURO1_ESC', 'VEH_ESC'), ('VEH_SS', 'VEH_ESC')]

plt.style.use('dark_background')

# VIOLIN PLOTS
for variable in variables:

    # Set your original significance level
    alpha = 0.05
    
    # Perform Kruskal-Wallis test
    kw_stat, kw_p_value = kruskal(*[data[data[category_column] == label][variable] for label in hue_order])
    
    # Check if Kruskal-Wallis test is significant
    if kw_p_value < alpha:
        # Perform Dunn post hoc test
        dunn_results = posthoc_dunn(data, val_col=variable, group_col=category_column)
        
        # Number of pairwise comparisons
        num_comparisons = len(hue_order) * (len(hue_order) - 1) / 2
        
        # Bonferroni-adjusted significance level
        bonferroni_alpha = alpha / num_comparisons

    plt.figure(figsize=(12, 8))

    # Modified variable name without underscores and with capitalized words
    modified_variable = variable.replace("_", " ").title()

    # Create violin plot with dodge
    ax = sns.violinplot(x=category_column, y=variable, hue=category_column, data=data, palette=colors,
                        order=hue_order, inner='quartile', alpha=0.2, dodge=False, legend=False)

    # Customize the color of percentile lines directly
    for line in ax.lines:
        line.set_color("white")

    # Add scatter dots with dodge
    sns.stripplot(x=category_column, y=variable, hue=category_column, data=data, palette=colors,
                  order=hue_order, jitter=True, dodge=False, size=8, ax=ax, zorder=1, legend=False)

    # Remove spines and trim
    ax.grid(False)

    # Customize x-axis labels to be in capital letters
    ax.set_xticklabels([label.replace("_", " ").title().upper() for label in hue_order])

    plt.style.use('dark_background')

    # Customize title
    plt.title(f'{modified_variable}', fontsize=17, fontweight='bold')

    previous_label = None
        
    for idx, (label, other_label) in enumerate(dunn_order):
        # Check if label is the same as the previous_label
        if label == previous_label:
            continue
        else:
            # Adjust the position of the text for better visibility
            x_pos = (hue_order.index(label) + hue_order.index(other_label)) / 1.5
            y_pos = max(data[data[category_column] == label][variable].max(),
                        data[data[category_column] == other_label][variable].max()) + 5.0
            
            # Introduce a small offset based on the loop iteration
            offset = idx * 0.0
            
            text = f'{label} vs. {other_label} p-value: {dunn_results.loc[label, other_label]:.2e}'
            plt.text(x_pos, y_pos + offset, text, fontsize=7, ha='right', va='top', rotation=0)
            
            # Update previous_label for the next iteration
            previous_label = label



    plt.xlabel(category_column.upper(), fontsize=12, fontweight='bold')
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(Violin_path + f'{variable}_violin_plot.png', bbox_inches='tight', dpi=800)
    plt.show()
    
    

