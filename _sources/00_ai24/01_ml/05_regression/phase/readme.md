# Phase Estimation

### Overview

Phase estimation in linear regression involves estimating the phase parameter ($\theta$) of a sinusoidal model, particularly when noise is present. The given model is:

$$ y = \sin(wx + \theta) + \pi $$

where:
- $ w $ is known (frequency).
- $ \theta $ is unknown (phase).
- $ \pi $ is noise with zero mean, typically assumed to be Gaussian.

### Key Concepts

1. **Sinusoidal Model**: The model $ y = \sin(wx + \theta) + \pi $ is a sinusoidal function with a phase shift $\theta$. The parameter $w$ determines the frequency of the sine wave.

2. **Noise ($\pi$)**: The noise added to the model is Gaussian with zero mean, which implies the errors in measurements are normally distributed around the true value.

3. **Phase ($\theta$)**: The phase shift $\theta$ is the parameter of interest that we need to estimate from the observed data.

4. **Linear Regression**: Although the model is nonlinear due to the sine function, it can be transformed into a linear regression problem to facilitate the estimation of $\theta$.

### Transforming to Linear Regression

To estimate $\theta$, we transform the model into a form suitable for linear regression. Consider the trigonometric identity for a sine function:

$$ \sin(A + B) = \sin(A)\cos(B) + \cos(A)\sin(B) $$

Applying this to our model:

$$ \sin(wx + \theta) = \sin(wx)\cos(\theta) + \cos(wx)\sin(\theta) $$

Thus, the model becomes:

$$ y = \sin(wx)\cos(\theta) + \cos(wx)\sin(\theta) + \pi $$

We can rewrite this as:

$$ y = a\sin(wx) + b\cos(wx) + \pi $$

where $a = \cos(\theta)$ and $b = \sin(\theta)$.

### Linear Regression Formulation

We now have a linear regression problem where:

$$ y = a\sin(wx) + b\cos(wx) + \pi $$

Defining:
- $ X_1 = \sin(wx) $
- $ X_2 = \cos(wx) $

The model can be expressed as:

$$ y = aX_1 + bX_2 + \pi $$

This is a standard linear regression model:

$$ y = \beta_1 X_1 + \beta_2 X_2 + \epsilon $$

where $\beta_1 = a$ and $\beta_2 = b$, and $\epsilon = \pi$ (the noise term).

### Estimating $\theta$

In this linear regression setup:
- We can estimate $\beta_1$ and $\beta_2$ using Ordinary Least Squares (OLS) regression.
- The estimates of $\beta_1$ and $\beta_2$ are the estimates of $a$ and $b$, respectively.

Once we have the estimates $\hat{\beta}_1$ and $\hat{\beta}_2$:
- $\hat{a} = \hat{\beta}_1 = \cos(\theta)$
- $\hat{b} = \hat{\beta}_2 = \sin(\theta)$

We can then determine $\theta$ using the following relationships:

$$ \theta = \tan^{-1}\left(\frac{\hat{b}}{\hat{a}}\right) $$

### Advantages and Disadvantages

**Advantages**:
- **Simplicity**: Linear regression is straightforward and computationally efficient.
- **Closed-form solution**: OLS provides a closed-form solution for the estimates.
- **Applicability**: The method is broadly applicable when the noise is Gaussian.

**Disadvantages**:
- **Nonlinearity Handling**: The initial model is nonlinear, requiring transformation.
- **Assumptions**: Assumes Gaussian noise, which may not hold in all cases.
- **Phase Ambiguity**: $\theta$ can have multiple valid values due to the periodic nature of the tangent function.

### Applications

- **Signal Processing**: Estimating phase in sinusoidal signals.
- **Communications**: Phase estimation in modulated signals.
- **Time Series Analysis**: Identifying periodic components and their phases.

### Summary

To estimate $\theta$ in the model $ y = \sin(wx + \theta) + \pi $, we transform the model into a linear regression problem by expressing the sine function in terms of linear combinations of sine and cosine components. Using OLS regression, we estimate the coefficients of these components, which directly relate to the phase parameter $\theta$. This method leverages the properties of linear regression and is particularly effective under the assumption of Gaussian noise.