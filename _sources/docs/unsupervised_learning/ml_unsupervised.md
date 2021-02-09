<!-- Global site tag (gtag.js) - Google Analytics -->

<script async src="https://www.googletagmanager.com/gtag/js?id=G-ZLMLLKHZE0"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-ZLMLLKHZE0');
</script>
(sec:unsupervised)=
# Unsupervised Learning

In Sec. [](sec:supervised), we discussed supervised learning tasks, for which datasets consist of input-output pairs, or data-label pairs. More often than not, however, we have data without labels and would like to extract information from such a dataset. Clustering problems fall in this category, for instance: We suspect that the data can be divided into different types, but we do not know which features distinguish these types.

Mathematically, we can think of the data $\mathbf{x}$ as samples that were drawn from a probability distribution $P(\mathbf{x})$. The unsupervised learning task is to implicitly represent this distribution with a model, for example represented by a neural network. The model can then be used to study properties of the distribution or to generate new 'artificial' data. The models we encounter in this chapter are thus also referred to as *generative models*. In general, unsupervised learning is conceptually more challenging than supervised learning. At the same time, unsupervised algorithms are highly desirable, since unlabelled data is much more abundant than labelled data. Moreover, we can in principle use a generative model for a classification task by learning the joint probability distribution of the data-label pair.

In this chapter, we will introduce three types of neural networks that are specific to unsupervised learning tasks: *Restricted Boltzmann machines*, *autoencoders*, and *generative adversarial networks*. Furthermore, we will discuss how the RNN introduced in the previous chapter can also be used for an unsupervised task.

