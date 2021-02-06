

# Clustering Algorithms: the example of $k$-means

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