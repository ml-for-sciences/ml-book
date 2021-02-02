(sec:structuring_data)=
# Structuring Data without Neural Networks


Deep learning with neural networks is very much at the forefront of the
recent renaissance in machine learning. However, machine learning is not
synonymous with neural networks. There is a wealth of machine learning
approaches without neural networks, and the boundary between them and
conventional statistical analysis is not always sharp.

It is a common misconception that neural network techniques would always
outperform these approaches. In fact, in some cases, a simple linear
method could achieve faster and better results. Even when we might
eventually want to use a deep network, simpler approaches may help to
understand the problem we are facing and the specificity of the data so
as to better formulate our machine learning strategy. In this chapter,
we shall explore machine learning approaches without the use of neural
networks. This will further allow us to introduce basic concepts and the
general form of a machine learning workflow.

## Principle component analysis


At the heart of any machine learning task is data. In order to choose
the most appropriate machine learning strategy, it is essential that we
understand the data we are working with. However, very often, we are
presented with a dataset containing many types of information, called
*features* of the data. Such a dataset is also described as being
high-dimensional. Techniques that extract information from such a
dataset are broadly summarised as *high-dimensional inference*. For
instance, we could be interested in predicting the progress of diabetes
in patients given features such as age, sex, body mass index, or average
blood pressure. Extremely high-dimensional data can occur in biology,
where we might want to compare gene expression pattern in cells. Given a
multitude of features, it is neither easy to visualise the data nor pick
out the most relevant information. This is where *principle component
analysis* (PCA) can be helpful.

Very briefly, PCA is a systematic way to find out which feature or
combination of features varies the most across the data samples. We can
think of PCA as approximating the data with a high-dimensional
ellipsoid, where the principal axes of this ellipsoid correspond to the
principal components. A feature, which is almost constant across the
samples, in other words has a very short principal axis, might not be
very useful. PCA then has two main applications: (1) It helps to
visualise the data in a low dimensional space and (2) it can reduce the
dimensionality of the input data to an amount that a more complex
algorithm can handle.

### PCA algorithm

Given a dataset of $m$ samples with $n$ data features, we can arrange
our data in the form of a $m$ by $n$ matrix $X$ where the element
$x_{ij}$ corresponds to the value of the $j$th data feature of the $i$th
sample. We will also use the *feature vector* ${x}_i$ for all the $n$
features of one sample $i=1,\ldots,m$. The vector ${x}_i$ can take
values in the *feature space*, for example ${x}_i \in \mathbb{R}^n$.
Going back to our diabetes example, we might have $10$ data features.
Furthermore if we are given information regarding $100$ patients, our
data matrix $X$ would have $100$ rows and $10$ columns.

The procedure to perform PCA can then be described as follows:

```{admonition} Principle Component Analysis

1.  Center the data by subtracting from each column the mean of that
    column,

    ```{math}
	{x}_i \mapsto {x}_{i} - \frac{1}{m} \sum_{i=1}^{m} {x}_{i}.
          %  x_{ij} \longrightarrow x_{ij} - \frac{1}{m} \sum_{i=1}^{m} x_{ij}.
	```
    This ensures that the mean of each data feature is zero.

2.  Form the $n$ by $n$ (unnormalised) covariance matrix
    ```{math}
	:label: eqn:PCA-Covariance-Matrix
	C = {X}^{T}{X} = \sum_{i=1}^{m} {x}_{i}{x}_{i}^{T}.
    ```

3.  Diagonalize the matrix to the form
    $C = {X}^{T}{X} = W\Lambda W^{T}$, where the columns of $W$ are the
    normalised eigenvectors, or principal components, and $\Lambda$ is a
    diagonal matrix containing the eigenvalues. It will be helpful to
    arrange the eigenvalues from largest to smallest.

4.  Pick the $l$ largest eigenvalues $\lambda_1, \dots \lambda_l$,
    $l\leq n$ and their corresponding eigenvectors
    ${v}_1 \dots {v}_l$. Construct the $n$ by $l$ matrix
    $\widetilde{W} = [{v}_1 \dots {v}_l]$.

5.  Dimensional reduction: Transform the data matrix as

    ```{math}
	:label: eqn:PCA-Dimensional-Reduction
            \widetilde{X} = X\widetilde{W}.
    ``` 
The transformed data
    matrix $\widetilde{X}$ now has dimensions $m$ by $l$.
```





We have thus reduced the dimensionality of the data from $n$ to $l$.
Notice that there are actually two things happening: First, of course,
we now only have $l$ data features. But second, the $l$ data features
are new features and not simply a selection of the original data.
Rather, they are a linear combination of them. Using our diabetes
example again, one of the “new” data features could be the sum of the
average blood pressure and the body mass index. These new features are
automatically extracted by the algorithm.

But why did we have to go through such an elaborate procedure to do this
instead of simply removing a couple of features? The reason is that we
want to maximize the *variance* in our data. We will give a precise
definition of the variance later in the chapter, but briefly the
variance just means the spread of the data. Using PCA, we have
essentially obtained $l$ “new” features which maximise the spread of the
data when plotted as a function of this feature. We illustrate this with
an example.

### Example

Let us consider a very simple dataset with just $2$ data features. We
have data, from the Iris dataset [^1], a well known dataset on 3
different species of flowers. We are given information about the petal
length and petal width. Since there are just $2$ features, it is easy to
visualise the data. In {numref}`fig:Iris-PCA`, we show how the data is
transformed under the PCA algorithm.

```{figure} ../_static/lecture_specific/structuring_data/Iris-PCA.png
:name: fig:Iris-PCA

**PCA on Iris Dataset.**
```

Notice that there is no dimensional reduction here since $l = n$. In
this case, the PCA algorithm amounts simply to a rotation of the
original data. However, it still produces $2$ new features which are
orthogonal linear combinations of the original features: petal length
and petal width, i.e. 

```{math}

\begin{split}
        w_1 &= 0.922 \times \textrm{Petal Length} + 0.388 \times \textrm{Petal Width}, \\
        w_2 &= -0.388 \times \textrm{Petal Length} + 0.922 \times \textrm{Petal Width}.
\end{split}
```

We see very clearly that the first new feature $w_1$
has a much larger variance than the second feature $w_2$. In fact, if we
are interested in distinguishing the three different species of flowers,
as in a classification task, its almost sufficient to use only the data
feature with the largest variance, $w_1$. This is the essence of (PCA)
dimensional reduction.

Finally, it is important to note that it is not always true that the
feature with the largest variance is the most relevant for the task and
it is possible to construct counter examples where the feature with the
least variance contains all the useful information. However, PCA is
often a good guiding principle and can yield interesting insights in the
data. Most importantly, it is also *interpretable*, i.e., not only does
it separate the data, but we also learn *which* linear combination of
features can achieve this. We will see that for many neural network
algorithms, in contrast, a lack of interpretability is a big issue.

## Kernel PCA

PCA performs a linear transformation on the data. However, there are
cases where such a transformation is unable to produce any meaningful
result. Consider for instance the fictitious dataset with $2$ classes
and $2$ data features as shown on the left of {numref}`fig:Kernel-PCA`. We see by naked eye that it should be
possible to separate this data well, for instance by the distance of the
datapoint from the origin, but it is also clear that a linear function
cannot be used to compute it. In this case, it can be helpful to
consider a non-linear extension of PCA, known as *kernel PCA*.

The basic idea of this method is to apply to the data
$\mathbf{x} \in \mathbb{R}^{n}$ a chosen non-linear vector-valued
transformation function $\mathbf{\Phi}(\mathbf{x})$ with

```{math}
:label: eqn:kernel-pca-transformation
    \mathbf{\Phi}: \mathbb{R}^{n} \rightarrow \mathbb{R}^{N},
```

which is a map from the original $n$-dimensional space (corresponding to the $n$
original data features) to a $N$-dimensional feature space. Kernel PCA
then simply involves performing the standard PCA on the transformed data
$\mathbf{\Phi}(\mathbf{x})$. Here, we will assume that the transformed data is
centered, i.e., 

```{math}
\sum_i \Phi(\mathbf{x}_i) = 0
```

to have simpler formulas.

```{figure} ../_static/lecture_specific/structuring_data/circles_pca_kpca.png
:name: fig:Kernel-PCA

**Kernel PCA versus PCA.**
```
In practice, when $N$ is large, it is not efficient or even possible to
explicitly perform the transformation $\mathbf{\Phi}$. Instead we can make
use of a method known as the kernel trick. Recall that in standard PCA,
the primary aim is to find the eigenvectors and eigenvalues of the
covariance matrix $C$ . In the case of kernel PCA, this matrix becomes

```{math}
C = \sum_{i=1}^{m} \mathbf{\Phi}(\mathbf{x}_{i})\mathbf{\Phi}(\mathbf{x}_{i})^T,
```

with the eigenvalue equation

```{math}
:label: eqn:pca-eigenvalue-equation
\sum_{i=1}^{m} \mathbf{\Phi}(\mathbf{x}_{i})\mathbf{\Phi}(\mathbf{x}_{i})^T \mathbf{v}_{j} = \lambda_{j}\mathbf{v}_{j}.
```

By writing the eigenvectors $\mathbf{v}_{j}$ as a linear combination of the transformed data features

```{math}
\mathbf{v}_{j} = \sum_{i=1}^{m} a_{ji}\mathbf{\Phi}(\mathbf{x}_{i}),
```

we see that finding the eigenvectors is equivalent to finding the coefficients
$a_{ji}$. On substituting this form back into Eq. [](eqn:pca-eigenvalue-equation), we find

```{math}
\sum_{i=1}^{m} \mathbf{\Phi}(\mathbf{x}_{i})\mathbf{\Phi}(\mathbf{x}_{i})^T \left[ \sum_{i=1}^{m} a_{ji}\mathbf{\Phi}(\mathbf{x}_{j}) \right] = \lambda_{j} \left[ \sum_{i=1}^{m} a_{ji}\mathbf{\Phi}(\mathbf{x}_{i}) \right].
```

By multiplying both sides of the equation by $\mathbf{\Phi}(\mathbf{x}_{k})^{T}$
we arrive at 

```{math}
:label: eqn:kernel-pca-eigen-equation}
    \begin{split}
        \sum_{i=1}^{m} \mathbf{\Phi}(\mathbf{x}_{k})^{T} \mathbf{\Phi}(\mathbf{x}_{i})\mathbf{\Phi}(\mathbf{x}_{i})^T \left[ \sum_{l=1}^{m} a_{jl}\mathbf{\Phi}(\mathbf{x}_{l}) \right] &= \lambda_{j} \mathbf{\Phi}(\mathbf{x}_{k})^{T} \left[ \sum_{l=1}^{m} a_{jl} \mathbf{\Phi}(\mathbf{x}_{l}) \right] \\
        \sum_{i=1}^{m} \left[ \mathbf{\Phi}(\mathbf{x}_{k})^{T} \mathbf{\Phi}(\mathbf{x}_{i}) \right]   \sum_{l=1}^{m} a_{jl} \left[ \mathbf{\Phi}(\mathbf{x}_{i})^T \mathbf{\Phi}(\mathbf{x}_{l}) \right] &= \lambda_{j} \sum_{l=1}^{m} a_{jl} \left[ \mathbf{\Phi}(\mathbf{x}_{k})^{T} \mathbf{\Phi}(\mathbf{x}_{l}) \right] \\
        \sum_{i=1}^{m} K(\mathbf{x}_{k},\mathbf{x}_{i})   \sum_{l=1}^{m} a_{jl} K(\mathbf{x}_{i},\mathbf{x}_{l}) &= \lambda_{j} \sum_{l=1}^{m} a_{jl} K(\mathbf{x}_{k},\mathbf{x}_{l}), 
    \end{split}
```

where $K(\mathbf{x},\mathbf{y}) = \mathbf{\Phi}(\mathbf{x})^{T} \mathbf{\Phi}(\mathbf{y})$ is known as
the *kernel*. Thus we see that if we directly specify the kernels we can
avoid explicit performing the transformation $\mathbf{\Phi}$. In matrix
form, we find the eigenvalue equation
$K^{2}\mathbf{a}_{j} = \lambda_{j} K \mathbf{a}_{j}$, which simplifies to

```{math}
K \mathbf{a}_{j} = \lambda_{j} \mathbf{a}_{j}.
```

Note that this simplification requires $\lambda_j \neq 0$, which will be the case for relevant
principle components. (If $\lambda_{j} = 0$, then the corresponding
eigenvectors would be irrelevant components to be discarded.) After
solving the above equation and obtaining the coefficients $a_{jl}$, the
kernel PCA transformation is then simply given by the overlap with the
eigenvectors $\mathbf{v}_{j}$, i.e.,

```{math}
\mathbf{x} \rightarrow \mathbf{\Phi}(\mathbf{x}) \rightarrow y_{j} = \mathbf{v}_{j}^{T}\mathbf{\Phi}(\mathbf{x}) = \sum_{i=1}^{m} a_{ji} \mathbf{\Phi}(\mathbf{x}_{i})^{T} \mathbf{\Phi}(\mathbf{x}) = \sum_{i=1}^{m} a_{ji} K(\mathbf{x}_{i},\mathbf{x}),
```

where once again the explicit $\mathbf{\Phi}$ transformation is avoided.

A common choice for the kernel is known as the radial basis function
kernel (RBF) defined by

```{math}
K_{\textrm{RBF}}(\mathbf{x},\mathbf{y}) = \exp\left( -\gamma \| \mathbf{x} - \mathbf{y}\|^{2} \right),
```

where $\gamma$ is a tunable parameter. Using the RBF kernel, we compare
the result of kernel PCA with that of standard PCA, as shown on the
right of {numref}`fig:Kernel-PCA`. It is clear that kernel PCA leads to a
meaningful separation of the data while standard PCA completely fails.


## t-SNE as a nonlinear visualization technique

We studied (kernel) PCA as an example for a method that reduces the
dimensionality of a dataset and makes features apparent by which data
points can be efficiently distinguished. Often, it is desirable to more
clearly cluster similar data points and visualize this clustering in a
low (two- or three-) dimensional space. We focus our attention on a
relatively recent algorithm (from 2008) that has proven very performant.
It goes by the name t-distributed stochastic neighborhood embedding
(t-SNE).

The basic idea is to think of the data (images, for instance) as objects
$\mathbf{x}_i$ in a very high-dimensional space and characterize their
relation by the Euclidean distance $||\mathbf{x}_i-\mathbf{x}_j||$ between them.
These pairwise distances are mapped to a probability distribution
$p_{ij}$. The same is done for the distances $||\mathbf{y}_i-\mathbf{y}_j||$ of
the images of the data points $\mathbf{y}_i$ in the target low-dimensional
space. Their probability distribution is denoted $q_{ij}$. The mapping
is optimized by changing the locations $\mathbf{y}_i$ so as to minimize the
distance between the two probability distributions. Let us substantiate
these words with formulas.

The probability distribution in the space of data points is given as the
symmetrized version (joint probability distribution)

```{math}
p_{ij}=\frac{p_{i|j}+p_{j|i}}{2}
```

of the conditional probabilities

```{math}
p_{j|i}=\frac{\mathrm{exp}\left(-||\mathbf{x}_i-\mathbf{x}_j||^2/2\sigma_i^2\right)}
{\sum_{k\neq i}\mathrm{exp}\left(-||\mathbf{x}_i-\mathbf{x}_k||^2/2\sigma_i^2\right)},
```
where the choice of variances $\sigma_i$ will be explained momentarily.
Distances are thus turned into a Gaussian distribution. Note that
$p_{j|i}\neq p_{i|j}$ while $p_{ji}= p_{ij}$.

The probability distribution in the target space is chosen to be a
Student t-distribution 

```{math}
q_{ij}=\frac{
(1+||\mathbf{y}_i-\mathbf{y}_j||^2)^{-1}
}{
\sum_{k\neq l}
(1+||\mathbf{y}_k-\mathbf{y}_l||^2)^{-1}
}.
```

This choice has several advantages: (i) it is symmetric upon
interchanging $i$ and $j$, (ii) it is numerically more efficiently
evaluated because there are no exponentials, (iii) it has 'fatter' tails
which helps to produce more meaningful maps in the lower dimensional
space.

```{figure} ../_static/lecture_specific/structuring_data/pca_tSNE.png
:name: fig:PCA-vs-tsne

**PCA vs.\ t-SNE** Application of both methods on 5000 samples from the MNIST handwritten digit dataset. We see that perfect clustering cannot be achieved with either method, but t-SNE delivers the much better result.
```

Let us now discuss the choice of $\sigma_i$. Intuitively, in dense
regions of the dataset, a smaller value of $\sigma_i$ is usually more
appropriate than in sparser regions, in order to resolve the distances
better. Any particular value of $\sigma_i$ induces a probability
distribution $P_i$ over all the other data points. This distribution has
an *entropy* (here we use the Shannon entropy, in general it is a
measure for the "uncertainty" represented by the distribution)

```{math}
H(P_i)=-\sum_j p_{j|i}\, \mathrm{log}_2 \,p_{j|i}.
```

The value of $H(P_i)$ increases as $\sigma_i$ increases, i.e., the more uncertainty
is added to the distances. The algorithm searches for the $\sigma_i$
that result in a $P_i$ with fixed perplexity

```{math}
\mathrm{Perp}(P_i)=2^{H(P_i)}.
```

The target value of the perplexity is chosen a priory and is the main parameter that controls the outcome of
the t-SNE algorithm. It can be interpreted as a smooth measure for the
effective number of neighbors. Typical values for the perplexity are
between 5 and 50.

Finally, we have to introduce a measure for the similarity between the
two probability distributions $p_{ij}$ and $q_{ij}$. This defines a
so-called *loss function*. Here, we choose the *Kullback-Leibler*
divergence

```{math}
:label: eqn:KL
L(\{\mathbf{y}_i\})=\sum_i\sum_jp_{ij}\mathrm{log}\frac{p_{ij}}{q_{ij}},
```

which we will frequently encounter during this lecture. The symmetrized $p_{ij}$ ensures that $\sum_j p_{ij}>1/(2n)$, so that
each data point makes a significant contribution to the cost function.
The minimization of $L(\{\mathbf{y}_i\})$ with respect to the positions
$\mathbf{y}_i$ can be achieved with a variety of methods. In the simplest
case it can be gradient descent, which we will discuss in more detail in
a later chapter. As the name suggests, it follows the direction of
largest gradient of the cost function to find the minimum. To this end
it is useful that these gradients can be calculated in a simple form

```{math}
\frac{\partial L}{\partial \mathbf{y}_i}
=4\sum_j (p_{ij}-q_{ij})(\mathbf{y}_i-\mathbf{y}_j)(1+||\mathbf{y}_i-\mathbf{y}_j||^2)^{-1}.
```

By now, t-SNE is implemented as standard in many packages. They involve
some extra tweaks that force points $\mathbf{y}_i$ to stay close together at
the initial steps of the optimization and create a lot of empty space.
This facilitates the moving of larger clusters in early stages of the
optimization until a globally good arrangement is found. If the dataset
is very high-dimensional it is advisable to perform an initial
dimensionality reduction (to somewhere between 10 and 100 dimensions,
for instance) with PCA before running t-SNE.

While t-SNE is a very powerful clustering technique, it has its
limitations. (i) The target dimension should be 2 or 3, for much larger
dimensions ansatz for $q_{ij}$ is not suitable. (ii) If the dataset is
intrinsically high-dimensional (so that also the PCA pre-processing
fails), t-SNE may not be a suitable technique. (iii) Due to the
stochastic nature of the optimization, results are not reproducible. The
result may end up looking very different when the algorithm is
initialized with some slightly different initial values for $\mathbf{y}_i$.

## Clustering algorithms: the example of $k$-means

All of PCA, kernel-PCA and t-SNE may or may not deliver a visualization
of the dataset, where clusters emerge. They all leave it to the observer
to identify these possible clusters. In this section, we want to
introduce an algorithm that actually clusters data, i.e., it will sort
any data point into one of $k$ clusters. Here the desired number of
clusters $k$ is fixed a priori by us. This is a weakness but may be
compensated by running the algorithm with different values of $k$ and
asses where the performance is best.

We will exemplify a simple clustering algorithm that goes by the name
$k$-means. The algorithm is iterative. The key idea is that data points
are assigned to clusters such that the squared distances between the
data points belonging to one cluster and the centroid of the cluster is
minimized. The centroid is defined as the arithmetic mean of all data
points in a cluster.

This description already suggests, that we will again minimize a loss
function (or maximize an expectation function, which just differs in the
overall sign from the loss function). Suppose we are given an assignment
of datapoints $\mathbf{x}_i$ to clusters $j=1,\cdots, k$ that is represented
by 

```{math}
w_{ij}=\begin{cases}
1,\qquad \mathbf{x}_i\text{ in cluster }j,\\
0,\qquad \mathbf{x}_i\text{ not in cluster }j.
\end{cases}
```

Then the loss function is given by

```{math}
L(\{\mathbf{x}_i\},\{w_{ij}\})=\sum_{i=1}^m\sum_{j=1}^k w_{ij}||\mathbf{x}_i-\mathbf{\mu}_j ||^2,
```

where 

```{math}
:label: eqn:centroids
\mathbf{\mu}_j=\frac{\sum_iw_{ij}\mathbf{x}_i}{\sum_iw_{ij}}.
```

Naturally, we want to minimize the loss function with respect to the
assignment $w_{ij}$. However, a change in this assignment also changes
$\mathbf{\mu}_j$. For this reason, it is natural to divide each update step
in two parts. The first part updates the $w_{ij}$ according to

```{math}
w_{ij}=\begin{cases}
1,\qquad \text{if } j=\mathrm{argmin}_l ||\mathbf{x}_i-\mathbf{\mu}_l||,\\
0,\qquad \text{else }.
\end{cases}
```

That means we attach each data point to the nearest cluster centroid. The second part is a recalculation of the centroids
according to Eq. [](eqn:centroids).

The algorithm is initialized by choosing at random $k$ distinct data
points as initial positions of the centroids. Then one repeats the above
two-part steps until convergence, i.e., until the $w_{ij}$ do not change
anymore.

In this algorithm we use the Euclidean distance measure $||\cdot ||$. It
is advisable to standardize the data such that each feature has mean
zero and standard deviation of one when average over all data points.
Otherwise (if some features are overall numerically smaller than
others), the differences in various features may be weighted very
differently by the algorithm.

Furthermore, the results depend on the initialization. One should re-run
the algorithm with a few different initializations to avoid running into
bad local minima.

Applications of $k$-means are manifold: in economy they include marked
segmentation, in science any classification problem such as that of
phases of matter, document clustering, image compression (color
reduction), etc.. In general it helps to build intuition about the data
at hand.



[^1]: <https://archive.ics.uci.edu/ml/datasets/iris>