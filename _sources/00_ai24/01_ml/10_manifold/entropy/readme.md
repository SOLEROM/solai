# Entropy

## General Overview
Entropy, in the context of information theory, is a measure of the unpredictability or randomness of a data source. It's a fundamental concept that quantifies the amount of uncertainty or surprise associated with random variables. Mathematically, the entropy $ H(X) $ of a discrete random variable $ X $ with probability mass function $ p(x) $ is defined as:

$$
H(X) = - \sum_{x \in X} p(x) \log p(x)
$$

Here, $ p(x) $ is the probability of the occurrence of $ x $. The logarithm is typically base 2, and the unit of entropy is bits.

## Real-World Example
Consider a fair coin toss. The entropy of this process can be calculated as follows:

$$
H(X) = -\left( \frac{1}{2} \log_2 \frac{1}{2} + \frac{1}{2} \log_2 \frac{1}{2} \right) = -\left( \frac{1}{2} \times (-1) + \frac{1}{2} \times (-1) \right) = 1 \text{ bit}
$$

In this case, each coin toss produces one bit of entropy because there are two equally likely outcomes.

# Coding Data

## General Overview
Coding data refers to the process of encoding information using fewer bits based on its entropy. The goal is to represent data as efficiently as possible, reducing the average length of messages. This is fundamental in data compression techniques.

## Real-World Example
Huffman coding is a common method used to compress data. Suppose we have a set of characters with their corresponding frequencies:

- A: 45%
- B: 13%
- C: 12%
- D: 16%
- E: 9%
- F: 5%

Using Huffman coding, we can assign shorter codes to more frequent characters and longer codes to less frequent characters, thus minimizing the average code length.

# Kullback-Leibler Divergence

## General Overview
Kullback-Leibler (KL) Divergence is a measure of how one probability distribution diverges from a second, expected probability distribution. It is not symmetric and hence not a true metric but is widely used to measure the difference between two distributions $ P $ and $ Q $.

The KL Divergence from $ Q $ to $ P $ is defined as:

$$
D_{KL}(P \parallel Q) = \sum_{x \in X} P(x) \log \frac{P(x)}{Q(x)}
$$

## Real-World Example
Consider two probability distributions for a binary random variable:

- $ P(Head) = 0.6, P(Tail) = 0.4 $
- $ Q(Head) = 0.5, Q(Tail) = 0.5 $

The KL Divergence from $ Q $ to $ P $ is:

$$
D_{KL}(P \parallel Q) = 0.6 \log \frac{0.6}{0.5} + 0.4 \log \frac{0.4}{0.5} \approx 0.047 \text{ bits}
$$

This value indicates that there's a small divergence between the true distribution $ P $ and the approximate distribution $ Q $.

