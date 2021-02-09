<!-- Global site tag (gtag.js) - Google Analytics -->

<script async src="https://www.googletagmanager.com/gtag/js?id=G-ZLMLLKHZE0"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-ZLMLLKHZE0');
</script>
(sec:linear-methods-for-supervised-learning)=
# Supervised Learning without Neural Networks

*Supervised learning* is the term for a machine learning task, where we
are given a dataset consisting of input-output pairs
$\lbrace(\mathbf{x}_{1}, y_{1}), \dots, (\mathbf{x}_{m}, y_{m})\rbrace$ and our
task is to \"learn\" a function which maps input to output
$f: \mathbf{x} \mapsto y$. Here we chose a vector-valued input $\mathbf{x}$ and
only a single real number as output $y$, but in principle also the
output can be vector valued. The output data that we have is called the
*ground truth* and sometimes also referred to as "labels" of the input.
In contrast to supervised learning, all algorithms presented so far were
unsupervised, because they just relied on input-data, without any ground
truth or output data.

Within the scope of supervised learning, there are two main types of
tasks: *Classification* and *Regression*. In a classification task, our
output $y$ is a discrete variable corresponding to a classification
category. An example of such a task would be to distinguish stars with a
planetary system (exoplanets) from those without given time series of
images of such objects. On the other hand, in a regression problem, the
output $y$ is a continuous number or vector. For example predicting the
quantity of rainfall based on meteorological data from the previous
days.

In this section, we first familiarize ourselves with linear methods for
achieving these tasks. Neural networks, in contrast, are a non-linear
method for supervised classification and regression tasks.
