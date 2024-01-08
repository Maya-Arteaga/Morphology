                                              Morphometrics

![UMAP_HDBSCAN_CELLS2](https://github.com/Maya-Arteaga/Morphology/assets/70504322/ccab3478-0ad0-4d50-9b80-f754a08a884b)


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
(Siegart, 2022). However, due to the heterogeneity of microglia and each study, selecting features that best characterize microglia for each situation 
becomes a challenge. Considering the high biological heterogeneity, we have developed a pipeline that allows, based on the four main methods of microglial 
analysis and using data analysis techniques, the selection of features that best highlight differences between study groups. This approach aims to showcase
the intricacy of their morphologies while categorizing them in a simple and unbiased manner.


Feature Selection

As not all calculated variables are helpful and many might introduce noise (from less important features) or lead to collinearity, and too few features might fail to capture the complexity of microglial morphology, special feature selection techniques were employed to choose the most important characteristics. This approach also enhances the robustness of our model, as different features are selected based on the specificities of each study. The Recursive Feature Elimination (RFE) algorithm was chosen for this purpose. RFE systematically eliminates a set of features at a time using cross-validation. This approach allowed us to utilize variables that best highlighted the differences between the study groups. The scikit-learn package (https://scikit-learn.org/) was used for implementation.


Dimensionality Reduction and Clustering
Uniform Manifold Approximation and Projection (UMAP) serves as a dimension reduction technique designed for general non-linear dimensionality reduction while preserving both local and global structures. The algorithm assumes that the data is uniformly distributed on a Riemannian manifold, where the Riemannian metric is locally constant, and the manifold is locally connected (https://github.com/lmcinnes/umap/). Note that various hyperparameters (n_neighbors and min_dist) were experimented with, and yet the resulting structure remained robust.

Subsequently, Hierarchical Density-Based Spatial Clustering of Applications with Noise (HDBSCAN) was employed for clustering purposes. HDBSCAN utilizes a density-based approach, making minimal implicit assumptions about the clusters. This non-parametric method seeks a cluster hierarchy shaped by the multivariate modes of the underlying distribution. The approach involves transforming the space based on density, constructing the cluster hierarchy, and extracting the clusters. Notably, it accounts for data noise, making HDBSCAN more robust to noise and outliers by excluding data that is not near the densities from the clusters (https://github.com/scikit-learn-contrib/hdbscan).
![image](https://github.com/Maya-Arteaga/Morphology/assets/70504322/b8354d62-8a13-4fdc-9963-385fe5c1d310)




Juan Pablo Maya Arteaga
