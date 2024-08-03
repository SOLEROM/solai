# logistics regression


```
    oLogRegCls = LogisticRegression(penalty = paramPenalty, tol = 5e-3, C = paramC, solver = 'saga', max_iter = 10_000)

    oLogRegCls = oLogRegCls.fit(mXTrain, vYTrain)

    ROC AUC Train  = roc_auc_score(vYTrain, oLogRegCls.predict_proba(mXTrain), multi_class = 'ovr')
    ROC AUC Test = roc_auc_score(vYTest, oLogRegCls.predict_proba(mXTest), multi_class = 'ovr')


```

```
   Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.linear_model import LogisticRegression
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = LogisticRegression(random_state=0).fit(X, y)
    >>> clf.predict(X[:2, :])
    array([0, 0])
    >>> clf.predict_proba(X[:2, :])
    array([[9.8...e-01, 1.8...e-02, 1.4...e-08],
           [9.7...e-01, 2.8...e-02, ...e-08]])
    >>> clf.score(X, y)
    0.97...

    ```

```
Parameters
    ----------
    penalty : {'l1', 'l2', 'elasticnet', None}, default='l2'
        Specify the norm of the penalty:

        - `None`: no penalty is added;
        - `'l2'`: add a L2 penalty term and it is the default choice;
        - `'l1'`: add a L1 penalty term;
        - `'elasticnet'`: both L1 and L2 penalty terms are added.

        .. warning::
           Some penalties may not work with some solvers. See the parameter
           `solver` below, to know the compatibility between the penalty and
           solver.

        .. versionadded:: 0.19
           l1 penalty with SAGA solver (allowing 'multinomial' + L1)

    dual : bool, default=False
        Dual (constrained) or primal (regularized, see also
        :ref:`this equation <regularized-logistic-loss>`) formulation. Dual formulation
        is only implemented for l2 penalty with liblinear solver. Prefer dual=False when
        n_samples > n_features.

    tol : float, default=1e-4
        Tolerance for stopping criteria.

    C : float, default=1.0
        Inverse of regularization strength; must be a positive float.
        Like in support vector machines, smaller values specify stronger
        regularization.

    fit_intercept : bool, default=True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.

    intercept_scaling : float, default=1
        Useful only when the solver 'liblinear' is used
        and self.fit_intercept is set to True. In this case, x becomes
        [x, self.intercept_scaling],
        i.e. a "synthetic" feature with constant value equal to
        intercept_scaling is appended to the instance vector.
        The intercept becomes ``intercept_scaling * synthetic_feature_weight``.

        Note! the synthetic feature weight is subject to l1/l2 regularization
        as all other features.
        To lessen the effect of regularization on synthetic feature weight
        (and therefore on the intercept) intercept_scaling has to be increased.

    class_weight : dict or 'balanced', default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

        .. versionadded:: 0.17
           *class_weight='balanced'*

    random_state : int, RandomState instance, default=None
        Used when ``solver`` == 'sag', 'saga' or 'liblinear' to shuffle the
        data. See :term:`Glossary <random_state>` for details.

    solver : {'lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'}, \
            default='lbfgs'

        Algorithm to use in the optimization problem. Default is 'lbfgs'.
        To choose a solver, you might want to consider the following aspects:

            - For small datasets, 'liblinear' is a good choice, whereas 'sag'
              and 'saga' are faster for large ones;
            - For multiclass problems, only 'newton-cg', 'sag', 'saga' and
              'lbfgs' handle multinomial loss;
            - 'liblinear' is limited to one-versus-rest schemes.
            - 'newton-cholesky' is a good choice for `n_samples` >> `n_features`,
              especially with one-hot encoded categorical features with rare
              categories. Note that it is limited to binary classification and the
              one-versus-rest reduction for multiclass classification. Be aware that
              the memory usage of this solver has a quadratic dependency on
              `n_features` because it explicitly computes the Hessian matrix.

        .. warning::
           The choice of the algorithm depends on the penalty chosen.
           Supported penalties by solver:

           - 'lbfgs'           -   ['l2', None]
           - 'liblinear'       -   ['l1', 'l2']
           - 'newton-cg'       -   ['l2', None]
           - 'newton-cholesky' -   ['l2', None]
           - 'sag'             -   ['l2', None]
           - 'saga'            -   ['elasticnet', 'l1', 'l2', None]

        .. note::
           'sag' and 'saga' fast convergence is only guaranteed on features
           with approximately the same scale. You can preprocess the data with
           a scaler from :mod:`sklearn.preprocessing`.

        .. seealso::
           Refer to the User Guide for more information regarding
           :class:`LogisticRegression` and more specifically the
           :ref:`Table <Logistic_regression>`
           summarizing solver/penalty supports.

        .. versionadded:: 0.17
           Stochastic Average Gradient descent solver.
        .. versionadded:: 0.19
           SAGA solver.
        .. versionchanged:: 0.22
            The default solver changed from 'liblinear' to 'lbfgs' in 0.22.
        .. versionadded:: 1.2
           newton-cholesky solver.

    max_iter : int, default=100
        Maximum number of iterations taken for the solvers to converge.

    multi_class : {'auto', 'ovr', 'multinomial'}, default='auto'
        If the option chosen is 'ovr', then a binary problem is fit for each
        label. For 'multinomial' the loss minimised is the multinomial loss fit
        across the entire probability distribution, *even when the data is
        binary*. 'multinomial' is unavailable when solver='liblinear'.
        'auto' selects 'ovr' if the data is binary, or if solver='liblinear',
        and otherwise selects 'multinomial'.

        .. versionadded:: 0.18
           Stochastic Average Gradient descent solver for 'multinomial' case.
        .. versionchanged:: 0.22
            Default changed from 'ovr' to 'auto' in 0.22.

    verbose : int, default=0
        For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity.

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        Useless for liblinear solver. See :term:`the Glossary <warm_start>`.

        .. versionadded:: 0.17
           *warm_start* to support *lbfgs*, *newton-cg*, *sag*, *saga* solvers.

    n_jobs : int, default=None
        Number of CPU cores used when parallelizing over classes if
        multi_class='ovr'". This parameter is ignored when the ``solver`` is
        set to 'liblinear' regardless of whether 'multi_class' is specified or
        not. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors.
        See :term:`Glossary <n_jobs>` for more details.

    l1_ratio : float, default=None
        The Elastic-Net mixing parameter, with ``0 <= l1_ratio <= 1``. Only
        used if ``penalty='elasticnet'``. Setting ``l1_ratio=0`` is equivalent
        to using ``penalty='l2'``, while setting ``l1_ratio=1`` is equivalent
        to using ``penalty='l1'``. For ``0 < l1_ratio <1``, the penalty is a
        combination of L1 and L2.

```