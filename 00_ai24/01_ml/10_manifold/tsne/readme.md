# TSNE  

(t-distributed Stochastic Neighbor Embedding)

t-SNE is a powerful and widely-used technique for visualizing high-dimensional data by embedding it in a lower-dimensional space, typically two or three dimensions. This visualization helps to reveal the underlying structure and patterns within the data that may not be apparent in its original high-dimensional form.

## Algorithm

t-SNE converts similarities between data points to joint probabilities and tries to minimize the Kullback-Leibler divergence between the joint probabilities of the low-dimensional embedding and the high-dimensional data. The key difference in t-SNE compared to other methods is that it uses a varying $\sigma_i$ for each data point, which adapts to the local density of the data.

### Step-by-Step Breakdown

1. **Compute Pairwise Similarities:**
   - For high-dimensional data, compute the probability that point $ i $ would pick point $ j $ as its neighbor:
     $$
     p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2 \sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2 \sigma_i^2)}
     $$
   - Define a symmetric similarity measure:
     $$
     p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}
     $$
   - The variance $ \sigma_i $ is selected so that the perplexity of the conditional distribution equals a predefined value.

2. **Low-Dimensional Mapping:**
   - In the low-dimensional space, compute a similar joint probability distribution:
     $$
     q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|y_k - y_l\|^2)^{-1}}
     $$

3. **Minimize the Kullback-Leibler Divergence:**
   - Minimize the divergence between the high-dimensional and low-dimensional joint probabilities:
     $$
     KL(P \| Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}
     $$

## Objective

The objective function in t-SNE is designed to match the pairwise similarities of the points in the high-dimensional space with those in the low-dimensional space. This is achieved by minimizing the Kullback-Leibler divergence between the distributions $ P $ and $ Q $.


## Remarks

- **Hyperparameters Matter:**
  - **Perplexity**: This can be thought of as a smooth measure of the effective number of neighbors. It's crucial for balancing the attention given to local versus global aspects of your data.
  - **Learning Rate**: An appropriate learning rate is essential to ensure the algorithm converges properly.

- **Interpretation Cautions:**
  - **Cluster Size**: The size of clusters in the t-SNE plot does not directly reflect the actual sizes of clusters in the high-dimensional space.
  - **Cluster Distances**: Distances between clusters in the t-SNE plot may not be meaningful and can be misleading.

- **Noise and Patterns:**
  - **Random Noise**: Noise in the data might not appear random in the t-SNE visualization, as the algorithm can sometimes highlight patterns within noise.
  - **Shape Dependence**: The shapes of clusters are heavily influenced by the chosen perplexity value and may appear random or arbitrary.
  - **Topology Analysis**: To understand the topological structure of the data, it is often useful to examine t-SNE plots with multiple perplexity values.

t-SNE is an effective tool for exploring and visualizing the underlying structure of complex datasets. However, it requires careful tuning of hyperparameters and cautious interpretation of the resulting visualizations to draw meaningful conclusions.
