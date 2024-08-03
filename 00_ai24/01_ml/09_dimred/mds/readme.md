# MDS

## Overview

Multidimensional Scaling (MDS) is a technique used in data analysis to visualize the level of similarity or dissimilarity of data points. It is particularly useful in reducing the dimensions of complex data sets to make them easier to analyze and interpret.

### Distance to Similarity

The core idea behind MDS is to convert a matrix of distances (dissimilarities) between pairs of objects into a configuration of points in a low-dimensional space, typically 2D or 3D, such that the distances between the points in this space correspond as closely as possible to the original distances.

### Centering Matrix

To center the matrix of dissimilarities, we use a centering matrix $ \mathbf{H} $, which is defined as:

$$ \mathbf{H} = \mathbf{I} - \frac{1}{n} \mathbf{1} \mathbf{1}^T $$

Here, $ \mathbf{I} $ is the identity matrix, $ n $ is the number of objects, and $ \mathbf{1} $ is a column vector of ones.

## Classical MDS

Classical MDS, also known as Torgerson Scaling or Principal Coordinates Analysis, aims to place each object in $ \mathbb{R}^p $ such that the distances between points are preserved as well as possible.

### Closed Form Solution

The objective of classical MDS has a closed-form solution and is equivalent to Kernel PCA. The steps are:

1. Compute the squared distances $ D_{ij}^2 $ between all pairs of objects.
2. Double center the distance matrix using the centering matrix $ \mathbf{H} $:

$$ \mathbf{B} = -\frac{1}{2} \mathbf{H} \mathbf{D}^2 \mathbf{H} $$

3. Perform eigenvalue decomposition on $ \mathbf{B} $:

$$ \mathbf{B} = \mathbf{V} \mathbf{\Lambda} \mathbf{V}^T $$

4. The coordinates of the objects in the low-dimensional space are given by:

$$ \mathbf{X} = \mathbf{V} \mathbf{\Lambda}^{1/2} $$

## Algorithm

The algorithm for classical MDS can be summarized as:

1. Compute the matrix of squared distances $ \mathbf{D}^2 $.
2. Apply double centering to $ \mathbf{D}^2 $ using the centering matrix $ \mathbf{H} $.
3. Perform eigenvalue decomposition on the centered matrix $ \mathbf{B} $.
4. Use the top $ p $ eigenvectors and corresponding eigenvalues to obtain the coordinates in the $ p $-dimensional space.
5. Rotate and center the configuration of points such that they are centered on the origin (0, 0).

### Metric MDS

Metric MDS extends classical MDS by allowing the use of different distance metrics, such as the geodesic distance.

### Example with Geodesic Metric

When using geodesic distances, MDS can be applied to uncover the underlying structure of the data based on these distances. This is particularly useful in cases where the data lies on a non-linear manifold.

### Practical Example

Consider a set of cities with pairwise distances between them. MDS can be used to create a map that approximates the true geographic locations of the cities based on the given distances.

