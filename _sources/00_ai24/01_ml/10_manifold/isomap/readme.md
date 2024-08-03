# isoMap

# IsoMap

IsoMap is a nonlinear dimensionality reduction method that extends the ideas of Multidimensional Scaling (MDS) to non-linear manifolds. It is particularly useful for preserving the intrinsic geometry of the data, even when the data lies on a curved manifold.

## Overview

IsoMap works by constructing a graph from the data points and estimating the geodesic distances (shortest path distances along the manifold) between points. These distances are then used to embed the data into a lower-dimensional space using classical MDS.

### Steps Involved in IsoMap

1. **Construct the neighborhood graph:** 
   - Build a graph where each data point is connected to its nearest neighbors.
2. **Compute shortest paths:** 
   - Use an algorithm like Dijkstra's or Floyd-Warshall to compute the shortest paths between all pairs of points in the graph, approximating the geodesic distances.
3. **Apply MDS:** 
   - Perform classical MDS on the distance matrix derived from the shortest paths to find the low-dimensional embedding of the data.

## Constructing the Graph

### Step 1: Building a Graph on the Distance Matrix

The distance matrix $D$ contains the pairwise Euclidean distances between data points. To construct the graph:

1. For each point $i$, find its $k$-nearest neighbors.
2. Connect point $i$ to each of its $k$-nearest neighbors with an edge weighted by their Euclidean distance.

### Step 2: Graph of the Shortest Path

Once the neighborhood graph is constructed, the shortest paths between all pairs of points are computed to approximate the geodesic distances on the manifold. This step can be visualized as converting the initial graph into a fully connected graph where the edge weights represent the shortest path distances.

### Step 3: Classic MDS

Apply classical MDS on the geodesic distance matrix. MDS will embed the high-dimensional data into a lower-dimensional space while preserving the distances as closely as possible. The output is a set of coordinates in the lower-dimensional space.

## Mathematical Formulation

1. **Distance Matrix $D$:**
   $$
   D_{ij} = \| x_i - x_j \|
   $$

2. **Neighborhood Graph $G$:**
   Constructed using the $k$-nearest neighbors.

3. **Geodesic Distance Matrix $D_G$:**
   $$
   D_G(i, j) = \text{shortest path distance between } i \text{ and } j \text{ in } G
   $$

4. **Classical MDS:**
   - Center the matrix:
     $$
     B = -\frac{1}{2} H D_G^2 H
     $$
     where $H = I - \frac{1}{n} \mathbf{1}\mathbf{1}^T$.

   - Perform eigendecomposition on $B$:
     $$
     B = V \Lambda V^T
     $$
     where $\Lambda$ contains the top $d$ eigenvalues, and $V$ the corresponding eigenvectors.

   - The embedding is given by:
     $$
     Y = V \Lambda^{1/2}
     $$

## Real-World Examples

### Manifold Learning
In practice, IsoMap is used in manifold learning tasks where data lies on a non-linear manifold. For example, consider the "Swiss roll" dataset, which is a 2D surface embedded in a higher-dimensional space. IsoMap can effectively unravel the roll to find a meaningful 2D representation of the data.

### Image Recognition
In image recognition, IsoMap can be applied to reduce the dimensionality of high-dimensional image data while preserving important structural information, making it easier for algorithms to process and classify the images.

