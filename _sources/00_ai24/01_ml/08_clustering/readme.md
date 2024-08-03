# Clustering

Clustering is an unsupervised machine learning technique used to group similar data points into clusters. Unlike supervised learning, clustering does not require labeled data. 

```
clustering:
    recommendation system
    targeting marketing
    customer segmentation
    find anomalies
    
```

## intro
* [intro](./clustering.md)

## param clustering

* [kmean clustering](./kmean/readme.md)
    * [lab clustering](./kmean/0057ClusteringKMeans.ipynb)
    * [colorSpaceExample](./kmean/colorSpaceExample.md)
    * [lab colorSpaceExample](./kmean/0058ClusteringKMeans.ipynb)

* [gmm clustering](./gmm/readme.md)
    * [lab demo gmm](./gmm/0059ClusteringGMM.ipynb)

### limitation of gmm/kmean:

* the number of cluster must be specified
* kmean has linear boundary
* gmm has quadratic boundary




![alt text](image-3.png)

## non param clustering
* [hierarchical clustering](./hierarchical/readme.md)
    * [lab demo Agglomerative Clustering ](./hierarchical/0060ClusteringHierarchical.ipynb)


* [density](./density/readme.md)
    * [lab demo DBSCAN](./density/0061ClusteringDBSCAN.ipynb)

* [hdbscan](./hdbscan/readme.md)
    * [lab demo HDBSCAN](./hdbscan/0062ClusteringHDBSCAN.ipynb)


