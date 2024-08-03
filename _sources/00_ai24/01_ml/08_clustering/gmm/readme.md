# GMM

* GMM => Gaussian Mixture Models

Gaussian Mixture Models (GMM) are a type of probabilistic model used for clustering that generalizes the k-means algorithm by using a mixture of Gaussian distributions. Let's dive deeper into the concepts and workings of GMM.

## Differences from k-means

- **K-means** assigns each data point to the nearest cluster based on distance, resulting in straight-line boundaries between clusters.
- **GMM**, on the other hand, uses probabilistic "soft" assignments, where each data point is assigned a probability of belonging to each cluster. This results in more flexible, curved cluster boundaries.

## Multivariate Gaussian Distribution

A multivariate Gaussian distribution is a generalization of the one-dimensional (univariate) normal distribution to higher dimensions. It is defined by a mean vector $\mu$ and a covariance matrix $\Sigma$.

The probability density function of a multivariate Gaussian distribution is given by:

$$
f(x | \mu, \Sigma) = \frac{1}{(2\pi)^{k/2} |\Sigma|^{1/2}} \exp \left( -\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu) \right)
$$

where:
- $x$ is a $k$-dimensional data point.
- $\mu$ is the mean vector.
- $\Sigma$ is the covariance matrix.
- $|\Sigma|$ is the determinant of the covariance matrix.

## Covariance

The covariance matrix $\Sigma$ represents the extent to which the dimensions vary from the mean with respect to each other. For a dataset $X$ with $n$ data points each having $k$ dimensions, the covariance matrix is calculated as:

$$
\Sigma = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \mu)(x_i - \mu)^T
$$

where $x_i$ is a data point and $\mu$ is the mean of the data points.

## Gaussian Mixture Model

A GMM assumes that the data points are generated from a mixture of several Gaussian distributions with unknown parameters. Each Gaussian distribution in the mixture represents a cluster.

### Model

A GMM with $K$ components is defined as:

$$
p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x | \mu_k, \Sigma_k)
$$

where:
- $\pi_k$ is the mixing coefficient for the $k$-th Gaussian component, representing the probability of selecting the $k$-th component.
- $\mathcal{N}(x | \mu_k, \Sigma_k)$ is the Gaussian distribution with mean $\mu_k$ and covariance $\Sigma_k$.

### Expectation-Maximization Algorithm

Since solving the likelihood function directly is difficult, we use the Expectation-Maximization (EM) algorithm to estimate the parameters.

1. **Expectation Step (E-step)**: Calculate the posterior probability that each data point belongs to each Gaussian component.
   
   $$ 
   \gamma_{i,k} = \frac{\pi_k \mathcal{N}(x_i | \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(x_i | \mu_j, \Sigma_j)} 
   $$

2. **Maximization Step (M-step)**: Update the parameters (means, covariances, and mixing coefficients) using the posterior probabilities.
   
   $$ 
   \mu_k = \frac{\sum_{i=1}^{n} \gamma_{i,k} x_i}{\sum_{i=1}^{n} \gamma_{i,k}} 
   $$

   $$ 
   \Sigma_k = \frac{\sum_{i=1}^{n} \gamma_{i,k} (x_i - \mu_k)(x_i - \mu_k)^T}{\sum_{i=1}^{n} \gamma_{i,k}} 
   $$

   $$ 
   \pi_k = \frac{1}{n} \sum_{i=1}^{n} \gamma_{i,k} 
   $$

Repeat the E-step and M-step until convergence.

## Hard Assignment vs Soft Assignment

### Hard Assignment

In hard assignment, each data point belongs to a single cluster. This can lead to biased estimations, especially when data points are near cluster boundaries.

### Soft Assignment

In soft assignment, each data point has a probability of belonging to each cluster. This probabilistic approach leads to better estimations and more flexible cluster boundaries.

## Example

Consider a dataset with points that form two distinct clusters. Using GMM, we can model this data as a mixture of two Gaussian distributions.

