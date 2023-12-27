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
# Loop through variables
for variable in variables:
    # Perform Kruskal-Wallis test
    kruskal_result = kruskal(*[data[data[category_column] == category][variable] for category in hue_order])

    # Perform Dunn test for pairwise comparisons
    dunn_results = posthoc_dunn(data, val_col=variable, group_col=category_column, p_adjust="holm")

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

    # Annotate the plot with Dunn test results for all pairwise comparisons
    for pair in dunn_order:
        p_value = dunn_results.loc[pair[0], pair[1]]
        x_position = hue_order.index(pair[0]) + (hue_order.index(pair[1]) - hue_order.index(pair[0])) / 2.0
        y_position = data[data[category_column] == pair[0]][variable].max()
        ax.annotate(f'{pair[0]} vs {pair[1]} p={p_value:.2e}', xy=(x_position, y_position), color='white', fontsize=3,ha='center')

    # Remove spines and trim
    ax.grid(False)

    # Customize x-axis labels to be in capital letters
    ax.set_xticklabels([label.replace("_", " ").title().upper() for label in hue_order])

    plt.style.use('dark_background')

    # Customize title
    plt.title(f'{modified_variable}', fontsize=17, fontweight='bold')

    plt.xlabel(category_column.upper(), fontsize=12, fontweight='bold')
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(Dunn_path + f'{variable}_violin_plot.png', bbox_inches='tight', dpi=800)
    plt.show()
