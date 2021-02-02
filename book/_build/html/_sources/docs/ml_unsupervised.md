(sec:unsupervised)=
# Unsupervised Learning

In Sec. [](sec:supervised), we discussed supervised learning tasks, for which datasets consist of input-output pairs, or data-label pairs. More often than not, however, we have data without labels and would like to extract information from such a dataset. Clustering problems fall in this category, for instance: We suspect that the data can be divided into different types, but we do not know which features distinguish these types.

Mathematically, we can think of the data $\mathbf{x}$ as samples that were drawn from a probability distribution $P(\mathbf{x})$. The unsupervised learning task is to implicitly represent this distribution with a model, for example represented by a neural network. The model can then be used to study properties of the distribution or to generate new 'artificial' data. The models we encounter in this chapter are thus also referred to as *generative models*. In general, unsupervised learning is conceptually more challenging than supervised learning. At the same time, unsupervised algorithms are highly desirable, since unlabelled data is much more abundant than labelled data. Moreover, we can in principle use a generative model for a classification task by learning the joint probability distribution of the data-label pair.

In this chapter, we will introduce three types of neural networks that are specific to unsupervised learning tasks: *Restricted Boltzmann machines*, *autoencoders*, and *generative adversarial networks*. Furthermore, we will discuss how the RNN introduced in the previous chapter can also be used for an unsupervised task.

## Restricted Boltzmann machine

*Restricted Boltzmann Machines* (RBM) are a class of generative stochastic neural networks. More specifically, given some (binary) input data $\mathbf{x}\in\{0,1\}^{n_v}$, an RBM can be trained to approximate the probability distribution of this input. Moreover, once the neural network is trained to approximate the distribution of the input, we can sample from the network, in other words we generate new instances from the learned probability distribution.

The RBM consists of two layers (see {numref}`fig:RBM`) of *binary units*. Each binary unit is a variable which can take the values $0$ or $1$. We call the first (input) layer visible and the second layer hidden. The visible layer with input variables $\lbrace v_{1}, v_{2}, \dots v_{n_{\mathrm{v}}}\rbrace$, which we collect in the vector $\mathbf{v}$, is connected to the hidden layer with variables $\{ h_{1}, h_{2}, \dots h_{n_{\mathrm{h}}}\}$, which we collect in the vector $\mathbf{h}$. The role of the hidden layer is to mediate correlations between the units of the visible layer. In contrast to the neural networks we have seen in the previous chapter, the hidden layer is not followed by an output layer. Instead, the RBM represents a probability distribution $P_{\text{rbm}}(\mathbf{v})$, which depends on variational parameters represented by the weights and biases of a neural network. The RBM, as illustrated by the graph in {numref}`fig:RBM`, is a special case of a network structure known as a Boltzmann machine with the restriction that a unit in the visible layer is only connected to hidden units and vice versa, hence the name *restricted* Boltzmann machine.

```{figure} ../_static/lecture_specific/unsupervised-ml/rbm.png
:name: fig:RBM

**Restricted Boltzmann machine.** Each of the three visible units and
five hidden units represents a variable that can take the values $\pm1$
and the connections between them represent the entries $W_{ij}$ of the
weight matrix that enters the energy function [](eqn:RBM-Energy).
```


The structure of the RBM is motivated from statistical physics: To each choice of the binary vectors $\mathbf{v}$ and $\mathbf{h}$, we assign a value we call the energy 
```{math}
:label: eqn:RBM-Energy
E(\mathbf{v},\mathbf{h}) = -\sum_{i}a_{i}v_{i} - \sum_{j}b_{j}h_{j} - \sum_{ij} v_{i}W_{ij}h_{j},
```

where the vectors $\mathbf{a}$, $\mathbf{b}$, and the matrix $W$ are the variational parameters of the model. Given the energy, the probability distribution over the configurations $(\mathbf{v}, \mathbf{h})$ is defined as
```{math}
:label: eqn:RBM-Joint-Probability

P_{\textrm{rbm}}(\mathbf{v},\mathbf{h}) = \frac{1}{Z}e^{-E(\mathbf{v},\mathbf{h})},
```
where
```{math}
:label: eqn:partition-function

Z = \sum_{\mathbf{v},\mathbf{h}} e^{-E(\mathbf{v},\mathbf{h})}
```
is a normalisation factor called the partition function. The sum in Eq. [](eqn:partition-function) runs over all binary vectors $\mathbf{v}$ and $\mathbf{h}$, i.e., vectors with entries $0$ or $1$. The probability that the model assigns to a visible vector $\mathbf{v}$ is then the marginal over the joint probability distribution Eq. [](eqn:RBM-Joint-Probability),
```{math}
:label: eqn:RBM-visible-probability}

P_{\textrm{rbm}}(\mathbf{v}) = \sum_{\mathbf{h}} P_{\textrm{rbm}}(\mathbf{v},\mathbf{h}) = \frac{1}{Z}\sum_{h}e^{-E(\mathbf{v},\mathbf{h})}.
```
As a result of the restriction, the visible units, with the hidden units fixed, are mutually independent: given a choice of the hidden units $\mathbf{h}$, we have an **independent** probability distribution for
**each** visible unit given by
```{math}
	P_{\textrm{rbm}}(v_{i} = 1 | \mathbf{h}) = \sigma(a_{i} + \sum_{j}W_{ij}h_{j}), \qquad i=1,\ldots, n_{\mathrm{v}},
``` 
where $\sigma(x) = 1/(1+e^{-x})$ is the sigmoid function. Similarly, with the visible units fixed, the individual hidden units are also mutually independent with the probability distribution
```{math}
:label: eqn:RBM-P(h|v)

P_{\textrm{rbm}}(h_{j} = 1 | \mathbf{v}) = \sigma(b_{j} + \sum_{i}v_{i}W_{ij})\qquad j=1,\ldots, n_{\mathrm{h}}. 
```
The visible (hidden) units can thus be interpreted as artificial neurons connected to the hidden (visible) units with sigmoid activation function and bias $\mathbf{a}$ ($\mathbf{b}$). A direct consequence of this mutual independence is that sampling a vector $\mathbf{v}$ or $\mathbf{h}$ reduces to sampling every component individually. Notice that this simplification comes about due to the restriction that visible (hidden) units do not directly interact amongst themselves, i.e. there are no terms proportional to $v_i v_j$ or $h_i h_j$ in Eq. [](eqn:RBM-Energy). In the following, we explain how one can train an RBM and discuss possible applications of RBMs.

### Training an RBM

Consider a set of binary input data $\mathbf{x}_k$, $k=1,\ldots,M$, drawn from a probability distribution $P_{\textrm{data}}(\mathbf{x})$. The aim of the training is to tune the parameters $\lbrace \mathbf{a}, \mathbf{b}, W \rbrace$ in an RBM such that after training $P_{\textrm{rbm}}(\mathbf{x}) \approx  P_{\textrm{data}}(\mathbf{x})$. The standard approach to solve this problem is the maximum likelihood principle, in other words we want to find the parameters $\lbrace \mathbf{a}, \mathbf{b}, W \rbrace$ which maximize the probability that our model produces the data $\mathbf{x}_k$.

Maximizing the likelihood $\mathcal{L}(\mathbf{a},\mathbf{b},W) =  \prod P_{\textrm{rbm}}(\mathbf{x}_{k})$ is equivalent to training the RBM using a loss function we have encountered before, the negative log-likelihood
```{math}
	L(\mathbf{a},\mathbf{b},W) = - \sum_{k=1}^{M} \log P_{\textrm{rbm}}(\mathbf{x}_{k}).
```
For the gradient descent, we need derivatives of the loss function of the form
```{math}
:label: eqn:log-likelihood-derivative

\frac{\partial L(\mathbf{a},\mathbf{b},W)}{\partial W_{ij}} = -\sum_{k=1}^{M} \frac{\partial\log P_{\textrm{rbm}}(\mathbf{x}_{k})}{\partial W_{ij}}.
```
This derivative consists of two terms, 
```{math}
:label: eqn:RBM-derivatives
\begin{split}
        \frac{\partial\log P_{\textrm{rbm}}(\mathbf{x})}{\partial W_{ij}} %&= \frac{\partial}{\partial W_{ij}}\left( -\log \sum_{\bm{vh}} e^{E(\mathbf{v},\mathbf{h})} + \log \sum_{\mathbf{h}} e^{E(\mathbf{x},\mathbf{h})}  \right) \\
        %&= -\frac{1}{Z}\sum_{\bm{vh}}v_{i}h_{j}e^{E(\mathbf{v},\mathbf{h})} +  \frac{1}{\sum_{\mathbf{h}}e^{E(\mathbf{x},\mathbf{h})}} \sum_{h}x_{i}h_{j}e^{E(\mathbf{x},\mathbf{h})}\\
        %&= \sum_{h_{j}}x_{i}h_{j} P_{\textrm{rbm}}(h_{j}|\mathbf{x}) - \sum_{\mathbf{v},\mathbf{h}} v_{i} h_{j} P_{\textrm{rbm}}(\mathbf{v},\mathbf{h}) \\
        &= x_{i}P_{\textrm{rbm}}(h_{j}=1|\mathbf{x}) - \sum_{\mathbf{v}} v_{i} P_{\textrm{rbm}}(h_{j}=1|\mathbf{v}) P_{\textrm{rbm}}(\mathbf{v})
\end{split}
```

and similarly simple forms are found for the derivatives with respect to the components of $\mathbf{a}$ and $\mathbf{b}$. We can then iteratively update the parameters just as we have done in Chapter [](sec:supervised),
```{math}
W_{ij} \rightarrow W_{ij} - \eta \frac{\partial L(a,b,W)}{\partial W_{ij}}
```
with a sufficiently small learning rate $\eta$. As we have seen in the previous chapter in the context of backpropagation, we can reduce the computational cost by replacing the summation over the whole data set in Eq. [](eqn:log-likelihood-derivative) with a summation over a small randomly chosen batch of samples. This reduction in the computational cost comes at the expense of noise, but at the same time it can help to improve generalization.

However, there is one more problem: The second summation in Eq. [](eqn:RBM-derivatives), which contains $2^{n_v}$ terms, cannot be efficiently evaluated exactly. Instead, we have to approximate the sum by sampling the visible layer $\mathbf{v}$ from the marginal probability distribution $P_{\textrm{rbm}}(\mathbf{v})$. This sampling can be done using *Gibbs sampling* as follows:

```{admonition} Gibbs-Sampling
:name: alg:Gibbs-Sampling
**Input:** Any visible vector $\mathbf{v}(0)$  <br />
**Output:** Visible vector $\mathbf{v}(r)$  <br />
**for:** $n=1$\dots $r$  <br />
$\quad$ sample $\mathbf{h}(n)$ from $P_{\rm rbm}(\mathbf{h}|\mathbf{v}=\mathbf{v}(n-1))$  <br />
$\quad$ sample $\mathbf{v}(n)$ from $P_{\rm rbm}(\mathbf{v}|\mathbf{h}=\mathbf{h}(n))$  <br />
**end** 
```

With sufficiently many steps $r$, the vector $\mathbf{v}(r)$ is an unbiased sample drawn from $P_{\textrm{rbm}}(\mathbf{v})$. By repeating the procedure, we can obtain multiple samples to estimate the summation. Note that this is still rather computationally expensive, requiring multiple evaluations on the model.

The key innovation which allows the training of an RBM to be computationally feasible was proposed by Geoffrey Hinton (2002). Instead of obtaining multiple samples, we simply perform the Gibbs sampling with $r$ steps and estimate the summation with a single sample, in other words we replace the second summation in Eq. [](eqn:RBM-derivatives) with
```{math}
\sum_{\mathbf{v}} v_{i} P_{\textrm{rbm}}(h_{j}=1|\mathbf{v}) P_{\textrm{rbm}}(\mathbf{v}) \rightarrow v'_{i} P_{\textrm{rbm}}(h_{j}=1|\mathbf{v}'),
```
where $\mathbf{v}' = \mathbf{v}(r)$ is simply the sample obtained from $r$-step Gibbs sampling. With this modification, the gradient, Eq. [](eqn:RBM-derivatives), can be approximated as
```{math}
\frac{\partial\log P_{\textrm{rbm}}(\mathbf{x})}{\partial W_{ij}} \approx x_{i}P_{\textrm{rbm}}(h_{j}=1|\mathbf{x}) -  v'_{i} P_{\textrm{rbm}}(h_{j}=1|\mathbf{v}').
```

This method is known as *contrastive divergence*. Although the quantity computed is only a biased estimator of the gradient, this approach is found to work well in practice. The complete algorithm for training a RBM with $r$-step contrastive divergence can be summarised as follows:

```{admonition} Contrastive divergence
:name: alg:contrastive-divergence
**Input:** Dataset $\mathcal{D} = \lbrace \ \mathbf{x}_{1}, \ \mathbf{x}_{2}, \dots \ \mathbf{x}_{M} \rbrace$ drawn from a distribution $P(x)$} <br />
initialize the RBM weights $\lbrace \mathbf{a},\mathbf{b},W \rbrace$ <br />
Initialize $\Delta W_{ij} = \Delta a_{i} = \Delta b_{j} =0$ <br />
**while:** not converged **do** <br />
$\quad$  select a random batch $S$ of samples from the dataset $\mathcal{D}$ <br />
$\quad$ **forall** $\mathbf{x} \in S$ <br />
$\quad\quad$ Obtain $\ \mathbf{v}'$ by $r$-step Gibbs sampling starting from $\ \mathbf{x}$ <br />
$\quad\quad$ $\Delta W_{ij} \leftarrow \Delta W_{ij} - x_{i}P_{\textrm{rbm}}(h_{j}=1|\ \mathbf{x}) +  v'_{i} P_{\textrm{rbm}}(h_{j}=1|\ \mathbf{h}')$ <br />
$\quad$ **end** <br />
$\quad$  $W_{ij} \leftarrow W_{ij} - \eta\Delta W_{ij}$ <br />
$\quad$ (and similarly for $\mathbf{a}$ and $\mathbf{b}$) <br />
**end** 
```


Having trained the RBM to represent the underlying data distribution
$P(\mathbf{x})$, there are a few ways one can use the trained model:

1.  **Pretraining:** We can use $W$ and $\mathbf{b}$ as the initial weights
    and biases for a deep network (c.f. Chapter 4), which is then
    fine-tuned with gradient descent and backpropagation.

2.  **Generative Modelling:** As a generative model, a trained RBM can be
    used to generate new samples via Gibbs sampling. Some potential uses of the
    generative aspect of the RBM include *recommender systems* and
    *image reconstruction*. In the following subsection, we provide an
    example, where an RBM is used to reconstruct a noisy signal.

### Example: signal or image reconstruction/denoising

A major drawback of the simple RBMs for their application is the fact that they only take binary data as input. As an example, we thus look at simple periodic waveforms with 60 sample points. In particular, we use sawtooth, sine, and square waveforms. In order to have quasi-continuous data, we use eight bits for each point, such that our signal can take values from 0 to 255. Finally, we generate samples to train with a small variation in the maximum value, the periodicity, as well as the center point of each waveform.

After training the RBM using the contrastive divergence algorithm, we now have a model which represents the data distribution of the binarized waveforms. Consider now a signal which has been corrupted, meaning some parts of the waveform have not been received, in other words they are set to 0. By feeding this corrupted data into the RBM and performing a few iterations of Gibbs sampling, we can obtain a reconstruction of the signal, where the missing part has been repaired, as can been seen at the bottom of {numref}`fig:RBM_reconstruction`.

Note that the same procedure can be used to reconstruct or denoise images. Due to the limitation to binary data, however, the picture has to either be binarized, or the input size to the RBM becomes fairly large for high-resolution pictures. It is thus not surprising that while RBMs have been popular in the mid-2000s, they have largely been superseded by more modern and architectures such as *generative adversarial networks* which we shall explore later in the chapter. However, they still serve a pedagogical purpose and could also provide inspiration for future innovations, in particular in science. A recent example is the idea of using an RBM to represent a quantum mechanical state.

```{figure} ../_static/lecture_specific/unsupervised-ml/rbm_reconstr.png
:name: fig:RBM_reconstruction

**Signal reconstruction.** Using an RBM to repair a corrupted signal,
here a sine and a sawtooth
waveform.
```

## Training an RNN without supervision


In Sec. [](sec:supervised), the RNN was introduced as a classification model. Instead of classifying sequences of data, such as time series, the RNN can also be trained to generate valid sequences itself. Given the RNN introduced in Sec. [](sec:rnn), the implementation of such a generator is straight-forward and does not require a new architecture. The main difference is that the output $\mathbf{y}_t$ of the network given the data point $\mathbf{x}_t$ is a guess of the subsequent data point $\mathbf{x}_{t+1}$ instead of the class to which the whole sequence belongs to. This means in particular that the input and output size are now the same. For training this network, we can once again use the cross-entropy or (negative) log-likelihood as a loss function, 
```{math}
:name: eqn:unsup_RNN
L_{\mathrm{ent}}
    =-\sum_{t=1}^{m-1} \mathbf{x}_{t+1}\cdot
    \ln \left(
    \mathbf{y}_{t}
    \right),
```
where $\mathbf{x}_{t+1}$ is now the 'label' for the input $\mathbf{x}_{t}$ and $\mathbf{y}_{t}$ is the output of the network and $t$ runs over the input sequence with length $m$. This training is schematically shown in {numref}`fig:RNN_gen`.

For generating a new sequence, it is enough to have one single input point $\mathbf{x}_1$ to start the sequence. Note that since we now can start with a single data point $\mathbf{x}_1$ and generate a whole sequence of data points $\{\mathbf{y}_t\}$, this mode of using an RNN is referred to as *one-to-many*. This sequence generation is shown in {numref}`fig:RNN_gen`, left.

```{figure} ../_static/lecture_specific/unsupervised-ml/generative_RNN2.png
:name: fig:RNN_gen

**RNN used as a generator.** For training, left, the input data
shifted by one, $\mathbf{x}_{t+1}$, are used as the label. For the
generation of new sequences, right, we input a single data point
$\mathbf{x}_1$ and the RNN uses the recurrent steps to generate a new
sequence.
```

(sec:rnn_gen)=
### Example: generating molecules with an RNN

To illustrate the concept of sequence generation using recurrent neural networks, we use an RNN to generate new molecules. The first question we need to address is how to encode a chemical structure into input data---of sequential form no less---that a machine learning model can read. A common representation of molecular graphs used in chemistry is the *simplified molecular-input line-entry system*, or SMILES. {numref}`fig:smiles` shows examples of such SMILES strings for the caffeine, ethanol, and aspirin molecules. We can use the dataset *Molecular Sets* [^1], which contains $\sim 1.9$M molecules written in the SMILES format.

Using the SMILES dataset, we create a dictionary to translate each character that appears in the dataset into an integer. We further use one-hot-encoding to feed each character separately to the RNN. This creates a map from characters in SMILES strings onto an array of numbers. Finally, in order to account for the variable size of the molecules and hence, the variable length of the strings, we can introduce a 'stop' character such that the network learns and later generates sequences of arbitrary length.

We are now ready to use the SMILES strings for training our network as described above, where the input is a one-hot-encoded vector and the output is again a vector of the same size. Note, however, that similar to a classification task, the output vector is a probability distribution over the characters the network believes could come next. Unlike a classification task, where we consider the largest output the best guess of the network, here we sample in each step from the probability distribution $\mathbf{y}_t$ to again have a one-hot-encoded vector for the input of the next step.


```{figure} ../_static/lecture_specific/unsupervised-ml/SMILES_examples.png
:name: fig:smiles

**SMILES.** Examples of molecules and their representation in
SMILES.
```
## Autoencoders

Autoencoders are neuron-based generative models, initially introduced for dimensionality reduction. The original purpose, thus, is similar to that of PCA or t-SNE that we already encountered in Sec. [](sec:structuring_data), namely the reduction of the number of features that describe our input data. Unlike for PCA, where we have a clear recipe how to reduce the number of features, an autoencoder learns the best way of achieving the dimensionality reduction. An obvious question, however, is how to measure the quality of the compression, which is essential for the definition of a loss function and thus, training. In the case of t-SNE, we introduced two probability distributions based on the distance of samples in the original and feature space, respectively, and minimized their difference, for example using the Kullback-Leibler divergence.

The solution the autoencoder uses is to have a neural network do first, the dimensionality reduction, or encoding to the *latent space*, $\mathbf{x}\mapsto \mathbf{e}(\mathbf{x})=\mathbf{z}$, and then, the decoding back to the original dimension, $\mathbf{z} \mapsto \mathbf{d}(\mathbf{z})$, see {numref}`fig:AE_scheme`. This architecture allows us to directly compare the original input $\mathbf{x}$ with the reconstructed output $\mathbf{d}(\mathbf{e}(\mathbf{x}))$, such that the autoencoder trains itself unsupervised by minimizing the difference. A good example of a loss function that achieves successful training and that we have encountered already several times is the cross entropy,
```{math}
L_{\rm ae} = - \sum_i \mathbf{x}_i \cdot \ln[ \mathbf{d}(\mathbf{e}(\mathbf{x}_i))].
```
In other words, we compare point-wise the difference between the input to the encoder with the decoder's output.

Intuitively, the latent space with its lower dimension presents a bottleneck for the information propagation from input to output. The goal of training is to find and keep the most relevant information for the reconstruction to be optimal. The latent space then corresponds to the reduced space in PCA and t-SNE. Note that much like in t-SNE but unlike in PCA, the new features are in general not independent.


```{figure} ../_static/lecture_specific/unsupervised-ml/autoencoder.png
:name: fig:AE_scheme

**General autoencoder architecture.** A neural network is used to
contract a compressed representation of the input in the latent space. A
second neural network is used to reconstruct the original
input.
```

### Variational autoencoders

A major problem of the approach introduced in the previous section is its tendency to overfitting. As an extreme example, a sufficiently complicated encoder-decoder pair could learn to map all data in the training set onto a single variable and back to the data. Such a network would indeed accomplish completely lossless compression and decompression. However, the network would not have extracted any useful information from the dataset and thus, would completely fail to compress and decompress previously unseen data. Moreover, as in the case of the dimensionality-reduction schemes discussed in Sec. [](sec:structuring_data), we would like to analyze the latent space images and extract new information about the data. Finally, we might also want to use the decoder part of the autoencoder as a generator for new data. For these reasons, it is essential that we combat overfitting as we have done in the previous chapters by regularization.

The question then becomes how one can effectively regularize the autoencoder. First, we need to analyze what properties we would like the latent space to fulfil. We can identify two main properties:

1.  If two input data points are close (according to some measure),
    their images in the latent space should also be close. We call this
    property *continuity*.

2.  Any point in the latent space should be mapped through the decoder
    onto a meaningful data point, a property we call *completeness*.

While there are principle ways to achieve regularization along similar paths as discussed in the previous section on supervised learning, we will discuss here a solution that is particularly useful as a generative model: the *variational autoencoder* (VAE).

```{figure} ../_static/lecture_specific/unsupervised-ml/vae.png
:name: fig:VAE

**Architecture of variational autoencoder.** Instead of outputting a
point $z$ in the latent space, the encoder provides a distribution
$N(\boldsymbol \mu, \boldsymbol \sigma)$, parametrized by the means $\boldsymbol \mu$ and the
standard deviations $\boldsymbol \sigma$. The input $\mathbf{z}$ for the decoder is
then drawn from $N(\boldsymbol \mu, \boldsymbol \sigma)$.
```

The idea behind VAEs is for the encoder to output not just an exact point $\mathbf{z}$ in the latent space, but a (factorized) Normal distribution of points, $\mathcal{N}(\boldsymbol \mu, \boldsymbol \sigma)$. In particular, the output of the encoder comprises two vectors, the first representing the means, $\boldsymbol \mu$, and the second the standard deviations, $\boldsymbol \sigma$. The input for the decoder is then sampled from this distribution, $\mathbf{z} \sim \mathcal{N}(\boldsymbol \mu, \boldsymbol \sigma)$, and the original input is reconstructed and compared to the original input for training. In addition to the standard loss function comparing input and output of the VAE, we further add a regularization term to the loss function such that the distributions from the encoder are close to a standard normal distribution $\mathcal{N}(\boldsymbol 0, \boldsymbol 1)$. Using the Kullback-Leibler divergence, Eq. [](eqn:KL), to measure the deviation from the standard normal distribution, the full loss function then reads 
```{math}
:label: eqn:loss_vae
\begin{aligned}
    L_{\rm vae} &= -\sum_i \mathbf{x}^{\rm in}_i \ln \mathbf{x}^{\rm out}_i + {\rm KL} (\mathcal{N}(\boldsymbol \mu_i, \boldsymbol \sigma_i)|| \mathcal{N}(\boldsymbol 0, \boldsymbol 1))\nonumber\\
    &= -\sum_i \mathbf{x}^{\rm in}_i \ln \mathbf{x}^{\rm out}_i + \frac12 \sum_k [\sigma_{i,k}^2 + \mu_{i,k}^2 -1 -2 \ln\sigma_{i,k}].
\end{aligned}
```
In this expression, the first term quantifies the reconstruction loss with $\mathbf{x}_i^{\rm in}$ the input to and $\mathbf{x}_i^{\rm out}$ the reconstructed data from the VAE. The second term is the regularization on the latent space for each input data point, which for two (diagonal) Normal distributions can be simplified, see second line of Eq. [](eqn:loss_vae). This procedure regularizes the training through the introduction of noise, similar to the dropout layer in Section [](sec:supervised). However, the regularization here not only generically increases generalization, but also enforces the desired structure in the latent space.

The structure of a VAE is shown in {numref}`fig:VAE`. By enforcing the mean and variance structure of the encoder output, the latent space fulfills the requirements outlined above. This type of structure can then serve as a generative model for many different data types: anything from human faces to complicated molecular structures. Hence, the variational autoencoder goes beyond extracting information from a dataset, but can be used for the scientific discovery. Note, finally, that the general structure of the variational autoencoder can be applied beyond the simple example above. As an example, a different distribution function can be enforced in the latent space other than the standard Normal distribution, or a different neural network can be used as encoder and decoder, such as a RNN.

## Generative adversarial networks

In this section, we will be a concerned with a type of generative neural network, the generative adversarial network (GAN), which gained a very high popularity in recent years. Before getting into the details about this method, we give a quick systematic overview over types of generative methods, to place GANs in proper relation to them [^2].

### Types of generative models

```{figure} ../_static/lecture_specific/unsupervised-ml/GenerativeTaxonomy.png
:name: fig:generative-taxonomy

**Maximum likelihood approaches to generative
modeling.**
```


We restrict ourselves to methods that are based on the *maximum likelihood principle*. The role of the model is to provide an estimate $p_{\text{model}}(\mathbf{x};\boldsymbol \theta)$ of a probability distribution parametrized by parameters $\boldsymbol \theta$. The likelihood is the probability that the model assigns to the training data

```{math}
\prod_{i=1}^mp_{\text{model}}(\mathbf{x}_{i};\boldsymbol \theta),
```

where $m$ is again the number of samples in the data $\{\mathbf{x}_i\}$. The goal is to choose the parameters $\boldsymbol \theta$ such as to maximize the likelihood.
Thanks to the equality 

```{math}
\begin{split}
\boldsymbol \theta^*=&\,\underset{\boldsymbol \theta}{\text{argmax}}\prod_{i=1}^mp_{\text{model}}(\mathbf{x}_{i};\boldsymbol \theta)\\
=&\,\underset{\boldsymbol \theta}{\text{argmax}}\sum_{i=1}^m\mathrm{log}\,p_{\text{model}}(\mathbf{x}_{i};\boldsymbol \theta)
\end{split}
```

we can just as well work with the sum of logarithms, which is easier to handle. As we explained previously (see section on t-SNE), the maximization is equivalent to the minimization of the cross-entropy between two probability distributions: the 'true' distribution $p_{\mathrm{data}}(\mathbf{x})$ from which the data has been drawn and $p_{\text{model}}(\mathbf{x};\boldsymbol \theta)$. While we do not have access to $p_{\mathrm{data}}(\mathbf{x})$ in principle, we estimate it empirically as a distribution peaked at the $m$ data points we have.

Methods can now be distinguished by the way $p_{\mathrm{model}}$ is defined and evaluated (see {numref}`fig:generative-taxonomy`). We differentiate between models that define $p_{\mathrm{data}}(\mathbf{x})$ *explicitly* through some functional form. They have the general advantage that maximization of the likelihood is rather straight-forward, since we have direct access to this function. The downside is that the functional forms are generically limiting the ability of the model to fit the data distribution or become computationally intractable.

Among those explicit density models, we can further distinguish between those that represent a computationally tractable density and those that do not. An example for tractable explicit density models are *fully visible belief networks* (FVBNs) that decompose the probability distribution over an $n$-dimensional vector $\mathbf{x}$ into a product of conditional probabilities

```{math}
p_{\mathrm{model}}(\mathbf{x})=\prod_{j=1}^n\, p_{\mathrm{model}}(x_j|x_1,\cdots,x_{j-1}).
```

We can already see that, once we use the model to draw new samples, this is done one entry of the vector $\mathbf{x}$ at a time (first $x_1$ is drawn, then, knowing it, $x_2$ is drawn etc.). This is computationally costly and not parallelizable but is useful for tasks that are anyway sequential (like generation of human speech, where the so-called WaveNet employs FVBNs).

Models that encode an explicit density, but require approximations to maximize the likelihood that can either be variational in nature or use stochastic methods. We have seen examples for either. Variational methods define a lower bound to the log likelihood which can be maximized

```{math}
\mathcal{L}(\mathbf{x};\boldsymbol \theta)\leq \mathrm{log}\,p_{\text{model}}(\mathbf{x};\boldsymbol \theta).
```

The algorithm produces a maximum value of the log-likelihood that is at least as high as the value for $\mathcal{L}$ obtained (when summed over all data points). Variational autoencoders belong to this category. Their most obvious shortcoming is that $\mathcal{L}(\mathbf{x};\boldsymbol \theta)$ may represent a very bad lower bound to the log-likelihood (and is in general not guaranteed to converge to it for infinite model size), so that the distribution represented by the model is very different from $p_{\mathrm{data}}$. Stochastic methods, in contrast, often rely on a Markov chain process: The model is defined by a probability $q(\mathbf{x}'|\mathbf{x})$ from which the current sample $\mathbf{x}'$ is drawn, which depends on the previously drawn sample $\mathbf{x}$ (but not any others). RBMs are an example for this. They have the advantage that there is some rigorously proven convergence to $p_{\text{model}}$ with large enough size of the RBM, but the convergence may be slow. Like with FVBNs, the drawing process is sequential and thus not easily parallelizable.

All these classes of models allow for explicit representations of the probability density function approximations. In contrast, for GANs and related models, there is only an indirect access to said probability density: The model allows us to sample from it. Naturally, this makes optimization potentially harder, but circumvents many of the other previously mentioned problems. In particular

-   GANs can generate samples in parallel

-   there are few restrictions on the form of the generator function (as
    compared to Boltzmann machines, for instance, which have a
    restricted form to make Markov chain sampling work)

-   no Markov chains are needed

-   no variational bound is needed and some GAN model families are known
    to be asymptotically consistent (meaning that for a large enough
    model they are approximations to any probability distribution).

GANs have been immensely successful in several application scenarios. Their superiority against other methods is, however, often assessed subjectively. Most of performance comparison have been in the field of image generation, and largely on the ImageNet database. Some of the standard tasks evaluated in this context are:

-   generate an image from a sentence or phrase that describes its
    content ("a blue flower")

-   generate realistic images from sketches

-   generate abstract maps from satellite photos

-   generate a high-resolution ("super-resolution") image from a lower
    resolution one

-   predict a next frame in a video.

As far as more science-related applications are concerned, GANs have
been used to

-   predict the impact of climate change on individual houses

-   generate new molecules that have been later synthesized.

In the light of these examples, it is of fundamental importance to understand that GANs enable (and excel at) problems with multi-modal outputs. That means the problems are such that a single input corresponds to many different 'correct' or 'likely' outputs. (In contrast to a mathematical function, which would always produce the same output.) This is important to keep in mind in particular in scientific applications, where we often search for *the one answer*. Only if that is not the case, GANs can actually play out their strengths.

Let us consider image super-resolution as an illustrative example: Conventional (deterministic) methods of increasing image resolution would necessarily lead to some blurring or artifacts, because the information that can be encoded in the finer pixel grid simply is not existent in the input data. A GAN, in contrast, will provide a possibility how a realistic image could have looked if it had been taken with higher resolution. This way they add information that may differ from the true scene of the image that was taken -- a process that is obviously not yielding a unique answer since many versions of the information added may correspond to a realistic image.

```{figure} ../_static/lecture_specific/unsupervised-ml/GAN.png
:name: fig:GAN_scheme

**Architecture of a GAN.**
```


### The working principle of GANs

The optimization of all neural network models we discussed so far was formulated as minimization of a cost function. For GANs, while such a formulation is a also possible, a much more illuminating perspective is viewing the GAN as a *game* between two players, the *generator* ($G$) and the *discriminator* ($D$), see {numref}`fig:GAN_scheme`. The role of $G$ is to generate from some random input $\mathbf{z}$ drawn from a simple distribution samples that could be mistaken from being drawn from $p_{\mathrm{data}}$. The task of $D$ is to classify its input as generated by $G$ or coming from the data. Training should improve the performance of both $D$ and $G$ at their respective tasks simultaneously. After training is completed, $G$ can be used to draw samples that closely resembles those drawn from $p_{\mathrm{data}}$. In summary 

```{math}
\begin{split}
D_{\boldsymbol \theta_D}&:\ \mathbf{x}\mapsto \text{binary true/false},\\
G_{\boldsymbol \theta_G}&:\ \mathbf{z}\mapsto \mathbf{x},
\end{split}
```
where we have also indicated the two sets of parameters on which the two functions depend: $\boldsymbol \theta_D$ and $\boldsymbol \theta_G$, respectively. The game is then defined by two cost functions. The discriminator wants to minimize $J_D(\boldsymbol \theta_D,\boldsymbol \theta_G)$ by only changing $\boldsymbol \theta_D$, while the generator $J_G(\boldsymbol \theta_D,\boldsymbol \theta_G)$ by only changing $\boldsymbol \theta_G$. So, each players cost depends on both their and the other players parameters, the latter of which cannot be controlled by the player. The solution to this game optimization problem is a (local) minimum, i.e., a point in $(\boldsymbol \theta_D,\boldsymbol \theta_G)$-space where $J_D(\boldsymbol \theta_D,\boldsymbol \theta_G)$ has a local minimum with respect to $\boldsymbol \theta_D$ and $J_G(\boldsymbol \theta_D,\boldsymbol \theta_G)$ has a local minimum with respect to $\boldsymbol \theta_G$. In game theory such a solution is called a Nash equilibrium. Let us now specify possible choices for the cost functions as well as for $D$ and $G$.

The most important requirement of $G$ is that it is differentiable. It thus can (in contrast to VAEs) not have discrete variables on the output layer. A typical representation is a deep (possibly convolutional) neural network. A popular Deep Conventional architecture is called DCGAN. Then $\boldsymbol \theta_G$ are the networks weights and biases. The input $\mathbf{z}$ is drawn from some simple prior distribution, e.g., the uniform distribution or a normal distribution. (The specific choice of this distribution is secondary, as long as we use the same during training and when we use the generator by itself.) It is important that $\mathbf{z}$ has at least as high a dimension as $\mathbf{x}$ if the full multi-dimensional $p_{\text{model}}$ is to be approximated. Otherwise the model will perform some sort of dimensional reduction. Several tweaks have also been used, such as feeding some components of $\mathbf{z}$ into a hidden instead of the input layer and adding noise to hidden layers.

The training proceeds in steps. At each step, a minibatch of $\mathbf{x}$ is drawn from the data set and a minibatch of $\mathbf{z}$ is sampled from the prior distribution. Using this, gradient descent-type updates are performed: One update of $\boldsymbol \theta_D$ using the gradient of $J_D(\boldsymbol \theta_D,\boldsymbol \theta_G)$ and one of $\boldsymbol \theta_G$ using the gradient of $J_G(\boldsymbol \theta_D,\boldsymbol \theta_G)$.

### The cost functions

For $D$, the cost function of choice is the cross-entropy as with standard binary classifiers that have sigmoid output. Given that the labels are '1' for data and '0' for $\mathbf{z}$ samples, it is simply

```{math}
J_D(\boldsymbol \theta_D,\boldsymbol \theta_G)
=-\frac{1}{2 N_1}\sum_i\,\log\,D(\mathbf{x}_i)-\frac{1}{2 N_2}\sum_j\log (1-D(G(\mathbf{z}_j))),
```
where the sums over $i$ and $j$ run over the respective minibatches, which contain $N_1$ and $N_2$ points.

For $G$ more variations of the cost functions have been explored. Maybe the most intuitive one is

```{math}
J_G(\boldsymbol \theta_D,\boldsymbol \theta_G)=-J_D(\boldsymbol \theta_D,\boldsymbol \theta_G),
```

which corresponds to the so-called *zero-sum* or *minmax* game. Its solution is formally given by

```{math}
:label: eqn:GAN-Minmax
\boldsymbol \theta_G^\star=\underset{\boldsymbol \theta_G}{\text{arg min}}\ \ \underset{\boldsymbol \theta_D}{\text{max}}
\left[-J_D(\boldsymbol \theta_D,\boldsymbol \theta_G)\right].
``` 
This form of the cost is convenient for theoretical analysis, because there is only a single target function, which helps drawing parallels to conventional optimization. However, other cost functions have been proven superior in practice. The reason is that minimization can get trapped very far from an equilibrium: When the discriminator manages to learn rejecting generator samples with high confidence, the gradient of the generator will be very small, making its optimization very hard.

Instead, we can use the cross-entropy also for the generator cost function (but this time from the generator's perspective)
```{math}
J_G(\boldsymbol \theta_D,\boldsymbol \theta_G)=-\frac{1}{2 N_2}\sum_j\log\, D(G(\mathbf{z}_j)).
```
Now the generator maximizes the probability of the discriminator being mistaken. This way, each player still has a strong gradient when the player is loosing the game. We observe that this version of $J_G(\boldsymbol \theta_D,\boldsymbol \theta_G)$ has no direct dependence of the training data. Of course, such a dependence is implicit via $D$, which has learned from the training data. This indirect dependence also acts like a regularizer, preventing overfitting: $G$ has no possibility to directly 'fit' its output to training data.

### Remarks

In closing this section, we comment on a few properties of GANs, which also mark frontiers for improvements. One global problem is that GANs are typically difficult to train: they require large training sets and are highly susceptible to hyper-parameter fluctuations. It is currently an active topic of research to compensate for this with the structural modification and novel loss function formulations.

#### Mode collapse 

This may describe one of the most obvious problems of GANs: it refers to a situation where $G$ does not explore the full space to which $\mathbf{x}$ belongs, but rather maps several inputs $\mathbf{z}$ to the same output. Mode collapse can be more or less severe. For instance a $G$ trained on generating images may always resort to certain fragments or patterns of images. A formal reason for mode collapse is when the simultaneous gradient descent gravitates towards a solution
```{math}
\boldsymbol \theta_G^\star=\underset{\boldsymbol \theta_D}{\text{arg max}}\ \ \underset{\boldsymbol \theta_G}{\text{min}}
\left[-J_D(\boldsymbol \theta_D,\boldsymbol \theta_G)\right],
```

instead of the order in Eq. [](eqn:GAN-Minmax). (A priori it is not clear which of the two solutions is closer to the algorithm's doing.) Note that the interchange of min and max in general corresponds to a different solution: It is now sufficient for $G$ to always produce one (and the same) output that is classified as data by $D$ with very high probability. Due to the mode collapse problem, GANs are not good at exploring ergodically the full space of possible outputs. They rather produce few very good possible outputs.

One strategy to fight mode collapse is called *minibatch features*. Instead of letting $D$ rate one sample at a time, a minibatch of real and generated samples is considered at once. It then detects whether the generated samples are unusually close to each other.

#### Arithmetics with GANs 

It has been demonstrated that GANs can do linear arithmetics with inputs to add or remove abstract features from the output. This has been demonstrated using a DCGAN trained on images of faces. The gender and the feature 'wearing glasses' can be added or subtracted and thus changed at will. Of course such a result is only empirical, there is no formal mathematical theory why it works.

#### Using GANs with labelled data 

It has been shown that, if (partially) labeled data is available, using the labels when training $D$ may improve the performance of $G$. In this constellation, $G$ has still the same task as before and does not interact with the labels. If data with $n$ classes exist, then $D$ will be constructed as a classifier for $(n+1)$ classes, where the extra class corresponds to 'fake' data that $D$ attributes to coming from $G$. If a data point has a label, then this label is used as a reference in the cost function. If a datapoint has no label, then the first $n$ outputs of $D$ are summed up.

#### One-sided label smoothing 

This technique is useful not only for the $D$ in GANs but also other binary classification problems with neural networks. Often we observe that classifiers give proper results, but show a too confident probability. This overshooting confidence can be counteracted by one-sided label smoothing. The idea is to simply replace the target value for the real examples with a value slightly less than 1, e.g., 0.9. This smoothes the distribution of the discriminator. Why do we only perform this off-set one-sided and not also give a small nonzero value $\beta$ to the fake samples target values? If we were to do this, the optimal function for $D$ is
```{math}
D^\star(\mathbf{x})=\frac{p_{\mathrm{data}}(\mathbf{x})+\beta p_{\mathrm{model}}(\mathbf{x})}{p_{\mathrm{data}}(\mathbf{x})+  p_{\mathrm{model}}(\mathbf{x})}.
```
Consider now a range of $\mathbf{x}$ for which $p_{\mathrm{data}}(\mathbf{x})$ is small but $p_{\mathrm{model}}(\mathbf{x})$ is large (a "spurious mode"). $D^\star(\mathbf{x})$ will have a peak near this spurious mode. This means $D$ reinforces incorrect behavior of $G$. This will encourage $G$ to reproduce samples that it already makes (irrespective of whether they are anything like real data).


[^1]: <https://github.com/molecularsets/moses>
[^2]: following "NIPS 2016 Tutorial: Generative Adversarial Netoworks"
    Ian Goodfellow, arXiv:1701.001160


