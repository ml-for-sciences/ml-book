
## Function Approximation

When the state-action space becomes very large, we face two problems:
First, we can not use tabular methods anymore, since we can not store
all values. Second and more important, even if we could store all the
values, the probability of visiting all state-action pairs with the
above algorithms becomes increasingly unlikely, in other words most
states will never be visited during training. Ideally, we should thus
identify states that are 'similar', assign them 'similar' value, and
choose 'similar' actions when in these states. This grouping of similar
states is exactly the kind of *generalization* we tried to achieve in
the previous sections. Not surprisingly, reinforcement learning is most
successful when combined with neural networks.

In particular, we can parametrize a value function
$\hat{v}_\pi(s; \theta)$ and try to find parameters $\theta$ such that
$\hat{v}_\pi(s; \theta) \approx v_\pi(s)$. This approximation can be
done using the supervised-learning methods encountered in the previous
sections, where the target, or label, is given by the new estimate. In
particular, we can use the *mean squared value error* to formulate a
gradient descent method for an update procedure analogous to
Eq. [](eqn:policy_evaluation_modelfree). Starting from a state $S$
and choosing an action $A$ according to the policy $\pi(a|S)$, we update
the parameters

```{math}
\theta^{[k+1]} = \theta^{[k]} + \alpha [R +\gamma \hat{v}_\pi(S'; \theta^{[k]}) - \hat{v}_\pi(S; \theta^{[k]}) ] \nabla \hat{v}_\pi (S;\theta^{[k]})
```

with $0< \alpha < 1$ again the learning rate. Note that, even though the
new estimate also depends on $\theta^{[k]}$, we only take the derivative
with respect to the old estimate. This method is thus referred to as
*semi-gradient method*. In an similar fashion, we can reformulate the
Sarsa algorithm introduced for generalized gradient iteration.

[^1]: We assume here an episodic task. At the very beginning of
    training, we may initialize the state-action value function
    randomly.

