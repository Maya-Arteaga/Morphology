                                              Morphometrics
![Figure2](https://github.com/Maya-Arteaga/Morphology/assets/70504322/c498a759-7cff-4317-ba91-7fa1a8c1521f)

Introduction 

Since its discovery, microglia has intrigued researchers with its ability to respond to external stimuli, addapting its morphology, 
migrating and accumulating at lesion sites (Del Rio-Hortega, 1932). Microglia is a highly dynamic cell involved in highly complex 
homeostatic processes (Sierra et al., 2014; Lier, J. 2020; McNamara et al., 2023). This heterogeneity gives microglia great sensitivity
to different brain microenvironments, enabling it to perform a myrriad of functions (Matejuk, 2020). Due to its responsiveness to 
microenvironments, microglia is a key player in understanding and addressing neurological diseases and central nervous system 
disorders (Li, 2018; Matejuk, 2020). Consequently, quantitative measurement of microglial morphology has been widely used in 
neuroscience (Reddaway, 2023), such as assessing neuroinflammatory states (Green, 2022).

However, despite the development of quantitative methods to measure microglial morphology, there is a lack of comparability between
methods, leading to discrepancies in the results obtained (Green, 2022; Reddaway, 2023). Moreover, for reliable inferences, a more
precise quantification of microglia is needed to capture the variability in its morphology for a more biological interpretation 
(Green, 2022; Reddaway, 2023).

Detecting subtle changes in microglial morphology throughout the spectrum would provide an early indication of its immediate responses to
local environmental signals, given its irritability to microenvironment cues (Stratoulias, 2019; Siegert, 2022). In this context, most 
analyses have been restricted in terms of the number or type of features used in their analyses. The most widely used methods for analyzing
microglial morphology are: 1) Fractal analysis; 2) Skeleton analysis; 3) Cell body area and perimeter; and 4) Sholl analysis (Green, 2022).

Furthermore, biases in effect size can occur due to the selection of cells within the chosen sample (single-cell analysis vs. full photomicrograph
analysis; Green, 2022). Reductions in effect size can occur when averaged in the dataset (Simpson's paradox). For example, 67-93% less effect size
has been reported using full photomicrograph analysis than in single-cell analysis, using the same metrics (Green, 2022). Therefore, although 
the use of full photomicrograph analysis is desirable for its fleetness, it is avoided due to its statistical consequences (Green, 2022). 
Conversely, single-cell analysis is more labor-intensive but statistically shows greater differences, making it more commonly employed (Green, 2022).

To capture the complexity of microglial morphology, it's essential to use the fewest variables that provide the maximum amount of information possible 
(Siegart, 2022). However, due to the heterogeneity of microglia and the particularities of each study, selecting features that best characterize microglia for each situation 
becomes a challenge. Considering the high biological heterogeneity, we have developed a pipeline that allows, based on the four main methods of microglial 
analysis and using data analysis techniques, the selection of features that best highlight differences between study groups. This approach aims to showcase
the intricacy of their morphologies while categorizing them in a simple and unbiased manner.



Morphometrics Analysis

From the complete binary photomicrographs, each cell was identified. Subsequently, classic morphometric features were calculated. For Skeleton analysis, the following measures were computed: total length of branches (in pixels), number of initial points (also known as cell processes, processes that emerge from the cell soma), number of junction points (points where branches subdivide), and number of endpoints (end points of branches). Cell body analysis yielded the calculation of area, perimeter, circularity (where 1 represents a perfect circle), Feret diameter (maximum caliper diameter, the longest distance between any two points along the object boundary), compactness (how closely an object packs its mass or area into a given space: area/perimeter), aspect ratio (width/height), orientation (angle in degrees), and eccentricity (major axis/minor axis). These same features were calculated for the entire cell (not just the cell soma or the cell skeleton). Fractal analysis involved calculating convex hulls (the smallest convex set of pixels that encloses a cell), and the same calculations performed in cell body analysis were applied. Sholl analysis included determining the number of Sholl circles (a circle circumscribing the cell was calculated, and circles with increasing radii were created with the centroid of the cell soma as the center), crossing processes (the number of times cell processes intersect the Sholl circles), and the max distance (the maximum distance between the centroid and the four vertices of the image).


![Figure_morphometrics_analysis](https://github.com/Maya-Arteaga/Morphology/assets/70504322/c91266f2-07d2-4a28-85b4-e092a6c8beca)




Feature Selection

As not all calculated variables are helpful and many might introduce noise (from less important features) or lead to collinearity, and too few features might fail to capture the complexity of microglial morphology, special feature selection techniques were employed to choose the most important characteristics. This approach also enhances the robustness of our model, as different features are selected based on the specificities of each study. The Recursive Feature Elimination (RFE) algorithm was chosen for this purpose. RFE systematically eliminates a set of features at a time using cross-validation. This approach allowed us to utilize variables that best highlighted the differences between the study groups. The scikit-learn package (https://scikit-learn.org/) was used for implementation.


![Figure_RFE](https://github.com/Maya-Arteaga/Morphology/assets/70504322/c50ff3f4-bd54-42e9-b8a5-cdf13ba0bafe)



Dimensionality Reduction and Clustering

Uniform Manifold Approximation and Projection (UMAP) serves as a dimension reduction technique designed for general non-linear dimensionality reduction while preserving both local and global structures. The algorithm assumes that the data is uniformly distributed on a Riemannian manifold, where the Riemannian metric is locally constant, and the manifold is locally connected (https://github.com/lmcinnes/umap/). Note that various hyperparameters (n_neighbors and min_dist) were experimented with, and yet the resulting structure remained robust.

![Figure_UMAP_trials](https://github.com/Maya-Arteaga/Morphology/assets/70504322/4dac335a-7a1a-4fbc-804b-950ec96b4eb9)


Subsequently, Hierarchical Density-Based Spatial Clustering of Applications with Noise (HDBSCAN) was employed for clustering purposes. HDBSCAN utilizes a density-based approach, making minimal implicit assumptions about the clusters. This non-parametric method seeks a cluster hierarchy shaped by the multivariate modes of the underlying distribution. The approach involves transforming the space based on density, constructing the cluster hierarchy, and extracting the clusters. Notably, it accounts for data noise, making HDBSCAN more robust to noise and outliers by excluding data that is not near the densities from the clusters (https://github.com/scikit-learn-contrib/hdbscan).

![UMAP_HDBSCAN_CELLS2](https://github.com/Maya-Arteaga/Morphology/assets/70504322/ccab3478-0ad0-4d50-9b80-f754a08a884b)

![Histogram_Clusters_10_0 01_order](https://github.com/Maya-Arteaga/Morphology/assets/70504322/0d680224-681e-4e12-b9ed-d03076d19c85)

In this case, we are dealing with five groups:

VEH SS: Saline solution as vehicle

ESC: Scopolamine (toxin)

CNEURO 0.1: Treatment at dosage 0.1

CNEURO 1.0: Treatment at dosage 1.0

CNEURO SS: Treatment in a control group

![Figure_pie_chart](https://github.com/Maya-Arteaga/Morphology/assets/70504322/5641aba8-031c-49ac-8915-a2a0c6e28c61)


Spatial Visualization

As each data point corresponds to a cell, following Dimensionality Reduction and Clustering, each cell was color-coded based on its cluster. This approach serves two main purposes: 1)It facilitates the confirmation of similarities among cells belonging to the same cluster in a practical manner; 2) It provides information about the spatial distribution of each clustered cell. This aspect could be particularly beneficial for conducting further spatial analyses while considering other structures, potentially unveiling previously unexplored patterns in the physiopathology of diseases. Four photomicrographs were combined to illustrate the points mentioned above.



![Figure_Spatial_Visualization](https://github.com/Maya-Arteaga/Morphology/assets/70504322/ffa50504-7b4d-451d-b38c-6c75bfdea2a4)


Juan Pablo Maya Arteaga
