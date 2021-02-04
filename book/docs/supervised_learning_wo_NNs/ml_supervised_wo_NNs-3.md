
# More than two classes: Logistic Regression

In the following, we are interested in the case of $p$ classes with
$p>2$. After the previous discussion, it seems natural for the output to
take the integer values $y = 1, \dots, p$. However, it turns out to be
helpful to use a different, so-called *one-hot encoding*. In this
encoding, the output $y$ is instead represented by the $p$-dimensional
unit vector in $y$ direction $\mathbf{e}^{(y)}$,

```{math}
:label: eqn:One-Hot-Encoding
    y \longrightarrow \mathbf{e}^{(y)} =
    \begin{bmatrix}
        e^{(y)}_1 \\
        \vdots \\
        e^{(y)}_y \\
        \vdots \\
        e^{(y)}_{p}
    \end{bmatrix}
    =
    \begin{bmatrix}
        0 \\
        \vdots \\
        1 \\
        \vdots \\
        0
    \end{bmatrix},
``` 

where $e^{(y)}_l = 1$ if $l = y$ and zero for all other $l=1,\ldots, p$. A main advantage of this encoding is that we are
not forced to choose a potentially biasing ordering of the classes as we
would when arranging them along the ray of integers.

A linear approach to this problem then again mirrors the case for linear
regression. We fit a multi-variate linear model,
Eq. [](eqn:Multivariate-Linear-Model), to the one-hot encoded
dataset $\lbrace(\mathbf{x}_{1}, \mathbf{e}^{(y_1)}), \dots, (\mathbf{x}_{m}, \mathbf{e}^{(y_m)})\rbrace$.
By minimising the RSS, Eq. [](eqn:RSS), we obtain the solution

```{math}
\hat{\beta} = (\widetilde{X}^{T}\widetilde{X})^{-1} \widetilde{X}^{T} Y,
```

where $Y$ is the $m$ by $p$ output matrix. The prediction given an input
$\mathbf{x}$ is then a $p$-dimensional vector
$\mathbf{f}(\mathbf{x}|\hat{\beta}) = \tilde{\mathbf{x}}^{T} \hat{\beta}$. On a
generic input $\mathbf{x}$, it is obvious that the components of this
prediction vector would be real valued, rather than being one of the
one-hot basis vectors. To obtain a class prediction
$F(\mathbf{x}|\hat{\beta}) = 1, \dots, p$, we simply take the index of the
largest component of that vector, i.e.,

```{math}
F(\mathbf{x}|\hat{\beta}) = \textrm{argmax}_{k} f_{k}(\mathbf{x}|\hat{\beta}).
```

The $\textrm{argmax}$ function is a non-linear function and is a first
example of what is referred to as *activation function*.

For numerical minimization, it is better to use a smooth activation
function. Such an activation function is given by the *softmax* function

```{math}
F_k(\mathbf{x}|\hat{\beta})= \frac{e^{-f_k(\mathbf{x}|\hat{\beta})}}{\sum_{k'=1}^pe^{-f_{k'}(\mathbf{x}|\hat{\beta})}}.
```

Importantly, the output of the softmax function is a probability
$P(y = k|\mathbf{x})$, since $\sum_k F_k(\mathbf{x}|\hat{\beta}) = 1$. This
extended linear model is referred to as *logistic regression* [^3].

The current linear approach based on classification of one-hot encoded
data generally works poorly when there are more than two classes. We
will see in the next chapter that relatively straightforward non-linear
extensions of this approach can lead to much better results.


[^3]: Note that the softmax function for two classes is the logistic
    function.


