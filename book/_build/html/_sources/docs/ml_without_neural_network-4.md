Linear classifiers and their extensions
---------------------------------------

### Binary classification and support vector machines

In a classification problem, the aim is to categorize the inputs into
one of a finite set of classes. Formulated as a supervised learning
task, the dataset again consists of input-output pairs, i.e.
$\lbrace({x}_{1}, y_{1}), \dots, ({x}_{m}, y_{m})\rbrace$ with
${x}\in \mathbb{R}^n$. However, unlike regression problems, the
output $y$ is a discrete integer number representing one of the classes.
In a binary classification problem, in other words a problem with only
two classes, it is natural to choose $y\in\{-1, 1\}$.

We have introduced linear regression in the previous section as a method
for supervised learning when the output is a real number. Here, we will
see how we can use the same model for a binary classification task. If
we look at the regression problem, we first note that geometrically
$$\label{eqn: Univariate Linear Model B}
     f(\boldsymbol{x}|{\beta}) = \beta_0 + \sum_{j=1}^{n} \beta_{j}x_{j} = 0$$
defines a hyperplane perpendicular to the vector with elements
$\beta_{j\geq1}$. If we fix the length $\sum_{j=1}^n \beta_j^2=1$, then
$f({x}|{\beta})$ measures the (signed) distance of ${x}$ to the
hyperplane with a sign depending on which side of the plane the point
${x}_i$ lies. To use this model as a classifier, we thus define
$$F({x}|{\beta}) = \sign f({x}|{\beta}),
  \label{eq:binaryclassifier}$$ which yields $\{+1, -1\}$. If the two
classes are (completely) linearly separable, then the goal of the
classification is to find a hyperplane that separates the two classes in
feature space. Specifically, we look for parameters ${\beta}$, such
that $$y_i \tilde{{x}}_i^T{\beta} > M, \quad \forall i,
  \label{eq:separable}$$ where $M$ is called the *margin*. The optimal
solution $\hat{{\beta}}$ then maximizes this margin. Note that
instead of fixing the norm of $\beta_{j\geq1}$ and maximizing $M$, it is
customary to minimize $\sum_{j=1}^n \beta_j^2$ setting $M=1$ in Eq. .

In most cases, the two classes are not completely separable. In order to
still find a good classifier, we allow some of the points ${x}_i$ to
lie within the margin or even on the wrong side of the hyperplane. For
this purpose, we rewrite the optimization constraint Eq.  to
$$y_i \tilde{{x}}_i^T{\beta} > (1-\xi_i), \textrm{with } \xi_i \geq 0, \quad \forall i.
  \label{eq:notseparable}$$ We can now define the optimization problem
as finding
$$\min_{{\beta},\{\xi_i\}} \frac12 \sum_{j=1}^{n} \beta_j^2 + C\sum_i \xi_i
  \label{eq:optimalclassifierbeta}$$ subject to the constraint Eq. .
Note that the second term with hyperparameter $C$ acts like a
regularizer, in particular a lasso regularizer. As we have seen in the
example of the previous section, such a regularizer tries to set as many
$\xi_i$ to zero as possible.

![[**Binary classification.**]{} Hyperplane separating the two classes
and margin $M$ of the linear binary classifier. The support vectors are
denoted by a circle around
them.[]{data-label="fig:svm"}](figures/SVM_overlap)

We can solve this constrained minimization problem by introducing
Lagrange multipliers $\alpha_i$ and $\mu_i$ and solving
$$\min_{\beta, \{\xi_i\}} \frac12 \sum_{j=1}^{n} \beta_j^2 + C\sum_i \xi_i - \sum_i \alpha_i [y_i \tilde{{x}}_i^T{\beta} - (1-\xi_i)] - \sum_i\mu_i\xi_i,
  \label{eq:svm_lagrange}$$ which yields the conditions
$$\begin{aligned}
  \beta_j &=& \sum_i \alpha_i y_i x_{ij},\label{eq:svm_beta}\\
  0 &=& \sum_i \alpha_i y_i\\
  \alpha_i &=& C-\mu_i, \quad \forall i.
\label{eq:svm_derivatives}\end{aligned}$$ It is numerically simpler to
solve the dual problem
$$\min_{\{\alpha_i\}} \frac12 \sum_{i,i'} \alpha_i \alpha_{i'} y_i y_{i'} {x}_i^T {x}_{i'} - \sum_i \alpha_i
  \label{eq:svm_dual}$$ subject to $\sum_i \alpha_i y_i =0$ and
$0\leq \alpha_i \leq C$ [^1]. Using Eq. , we can reexpress $\beta_j$ to
find
$$f({x}|\{\alpha_i\}) = \sum_i{}' \alpha_i y_i {x}^T {x}_i + \beta_0,
  \label{eq:svm_f}$$ where the sum only runs over the points ${x}_i$,
which lie within the margin, as all other points have $\alpha_i\equiv0$
\[see Eq. \]. These points are thus called the *support vectors* and are
denoted in Fig. \[fig:svm\] with a circle around them. Finally, note
that we can use Eq.  again to find $\beta_0$.

**The Kernel trick and support vector machines**\
We have seen in our discussion of PCA that most data is not separable
linearly. However, we have also seen how the kernel trick can help us in
such situations. In particular, we have seen how a non-linear function
${\Phi}({x})$, which we first apply to the data ${x}$, can help
us separate data that is not linearly separable. Importantly, we never
actually use the non-linear function ${\Phi}({x})$, but only the
kernel. Looking at the dual optimization problem Eq.  and the resulting
classifier Eq. , we see that, as in the case of Kernel PCA, only the
kernel $K({x}, {y}) = {\Phi}({x})^T{\Phi}({y})$
enters, simplifying the problem. This non-linear extension of the binary
classifier is called a *support vector machine*.

### More than two classes: logistic regression

In the following, we are interested in the case of $p$ classes with
$p>2$. After the previous discussion, it seems natural for the output to
take the integer values $y = 1, \dots, p$. However, it turns out to be
helpful to use a different, so-called *one-hot encoding*. In this
encoding, the output $y$ is instead represented by the $p$-dimensional
unit vector in $y$ direction ${e}^{(y)}$,
$$\label{eqn: One-Hot Encoding}
    y \longrightarrow {e}^{(y)} =
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
    \end{bmatrix},$$ where $e^{(y)}_l = 1$ if $l = y$ and zero for all
other $l=1,\ldots, p$. A main advantage of this encoding is that we are
not forced to choose a potentially biasing ordering of the classes as we
would when arranging them along the ray of integers.

A linear approach to this problem then again mirrors the case for linear
regression. We fit a multi-variate linear model, Eq. , to the one-hot
encoded dataset
$\lbrace({x}_{1}, {e}^{(y_1)}), \dots, ({x}_{m}, {e}^{(y_m)})\rbrace$.
By minimising the RSS, Eq. , we obtain the solution
$$\hat{\beta} = (\widetilde{X}^{T}\widetilde{X})^{-1} \widetilde{X}^{T} Y,$$
where $Y$ is the $m$ by $p$ output matrix. The prediction given an input
${x}$ is then a $p$-dimensional vector
${f}({x}|\hat{\beta}) = \tilde{{x}}^{T} \hat{\beta}$. On a
generic input ${x}$, it is obvious that the components of this
prediction vector would be real valued, rather than being one of the
one-hot basis vectors. To obtain a class prediction
$F({x}|\hat{\beta}) = 1, \dots, p$, we simply take the index of the
largest component of that vector, i.e.,
$$F({x}|\hat{\beta}) = \textrm{argmax}_{k} f_{k}({x}|\hat{\beta}).$$
The $\textrm{argmax}$ function is a non-linear function and is a first
example of what is referred to as *activation function*.

For numerical minimization, it is better to use a smooth activation
function. Such an activation function is given by the *softmax* function
$$F_k({x}|\hat{\beta})= \frac{e^{-f_k({x}|\hat{\beta})}}{\sum_{k'=1}^pe^{-f_{k'}({x}|\hat{\beta})}}.$$
Importantly, the output of the softmax function is a probability
$P(y = k|{x})$, since $\sum_k F_k({x}|\hat{\beta}) = 1$. This
extended linear model is referred to as *logistic regression* [^2].

The current linear approach based on classification of one-hot encoded
data generally works poorly when there are more than two classes. We
will see in the next chapter that relatively straightforward non-linear
extensions of this approach can lead to much better results.

[^1]: Note that the constraints for the minimization are not equalities,
    but actually inequalities. A solution thus has to fulfil the
    additional Karush-Kuhn-Tucker constraints $$\begin{aligned}
        \alpha_i [y_i \tilde{{x}}_i^T{\beta} - (1-\xi_i)]&=&0,\label{eq:firstKKT}\\
        \mu_i\xi_i &=& 0,\\
        y_i \tilde{{x}}_i^T{\beta} - (1-\xi_i)&>& 0.
      \end{aligned}$$

[^2]: Note that the softmax function for two classes is the logistic
    function.
