# Umap

* Uniform Manifold Approximation and Projection
* https://umap-learn.readthedocs.io/en/latest/


## about

UMAP is a modern manifold learning technique for dimensionality reduction, similar to t-SNE but often faster and more scalable. It's particularly effective for large datasets and preserves more of the global structure of the data compared to t-SNE, making it useful for a wide range of applications from visualization to unsupervised learning tasks.

## how


UMAP operates under the assumption that the data is uniformly distributed on a locally connected Riemannian manifold and that the Riemannian metric is locally constant (or can be approximated as such). It starts by constructing a high-dimensional graph representing the data where edges between data points reflect the local neighborhood structure. It then optimizes a low-dimensional graph to be as structurally similar as possible, using a fuzzy set approach to compare relationships between points rather than purely geometric distances.

## params

* n_neighbors: Controls how UMAP balances local versus global structure in the data. It determines the size of the local neighborhood used to construct the initial high-dimensional graph. A smaller value emphasizes local structure, while a larger value can capture more of the global structure.

* n_components: The number of dimensions to project the data onto.

* min_dist: The minimum distance apart that points are allowed to be in the low-dimensional representation. Smaller values will result in tighter clusters.