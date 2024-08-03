# Non-Linear Least Squares (NLLS)

#### Overview
Non-Linear Least Squares (NLLS) is a method for estimating the parameters of a model when the relationship between the dependent and independent variables is non-linear. Unlike Ordinary Least Squares (OLS), which deals with linear relationships, NLLS is used when the model's equation is a non-linear function of the parameters.

#### Key Concepts

1. **Non-Linear Models**:
    - A non-linear model can be expressed as:
    $$
    y_i = f(x_i, \beta) + \epsilon_i
    $$
    where $y_i$ is the observed value, $x_i$ represents the independent variables, $\beta$ are the parameters to be estimated, and $\epsilon_i$ is the error term.

2. **Objective Function**:
    - The objective of NLLS is to minimize the sum of squared residuals:
    $$
    \text{Minimize} \sum_{i=1}^{n} (y_i - f(x_i, \beta))^2
    $$

3. **Iterative Optimization**:
    - Since the relationship is non-linear, closed-form solutions like those in OLS are not available. Instead, iterative numerical methods are used to find the parameter estimates that minimize the sum of squared residuals.

4. **Gradient-Based Methods**:
    - Common methods include:
      - **Gauss-Newton Method**: Approximates the function using a linear Taylor expansion and iteratively updates the parameters.
      - **Levenberg-Marquardt Algorithm**: Combines the Gauss-Newton method and gradient descent, offering a robust approach to handling the non-linear nature and convergence issues.

#### Applications

1. **Curve Fitting**:
    - NLLS is widely used in curve fitting where the model represents complex phenomena that cannot be captured by linear equations.

2. **Econometrics and Finance**:
    - Modeling economic growth, option pricing, and other financial instruments often requires non-linear models.

3. **Physics and Engineering**:
    - Many physical processes, such as radioactive decay, population growth, and mechanical systems, are inherently non-linear and require NLLS for accurate modeling.

4. **Biostatistics**:
    - Modeling growth rates of organisms, dose-response relationships, and other biological processes often involves non-linear models.

#### Advantages

1. **Model Flexibility**:
    - NLLS allows for a broader range of models, accommodating complex relationships between variables that linear models cannot capture.

2. **Better Fit**:
    - For inherently non-linear relationships, NLLS can provide a much better fit to the data, leading to more accurate predictions and insights.

3. **Real-World Applicability**:
    - Many real-world processes are non-linear, making NLLS a critical tool for accurately capturing and understanding these processes.

#### Disadvantages

1. **Computational Complexity**:
    - NLLS requires iterative algorithms, which can be computationally intensive, especially for large datasets or complex models.

2. **Convergence Issues**:
    - The optimization process can encounter convergence problems, such as local minima, slow convergence, or divergence, depending on the initial parameter estimates and the nature of the function.

3. **Parameter Sensitivity**:
    - The estimates of the parameters can be highly sensitive to the initial values chosen, necessitating good initial guesses to ensure proper convergence.

4. **Interpretation**:
    - Non-linear models can be more difficult to interpret compared to linear models, as the relationships between variables are not straightforward.

### Conclusion
Non-Linear Least Squares is a crucial technique for modeling and estimating parameters in systems where relationships are inherently non-linear. It provides flexibility and better fit for complex data but comes with challenges such as computational complexity and potential convergence issues. Despite these challenges, NLLS remains an essential tool in various fields, from economics and finance to physics and engineering, where non-linear relationships are prevalent.

## notes

* useful for physical models;

* Most solvers work on the idea: 
    * Set the movement direction. 
    * Solve the 1D problem: How far to move a long the direction, using some modeling of  the 1D function.

* In practice, use solvers written by experts as numerical issues and optimized calculations are the key.

* Try being expert on useful transformations for the data
    * Use Log to transfer exponential models to linear models.
    * Use Exp to transfer logarithmic models into linear models.
    * Trigonometric identities.
    * Convex approximation of  the problem.
    * When using such transformations, take into account the effect on the noise model.

* Since the problem is not convex and sensitive to the starting point, global optimization methods are used.

* Usually, we’ll use Non Linear Least Squares for parameter search. For prediction (Interpolation) it is better 
to use other methods