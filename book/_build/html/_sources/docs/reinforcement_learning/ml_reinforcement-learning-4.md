
# Temporal-difference Learning

If we cannot explicitly solve the Bellman optimality equations---the
case most often encountered---then we need to find the optimal policy by
some other means. If the state space is still small enough to keep track
of all value functions, we can tabulate the value function for all the
states and a given policy and thus, speak of *tabular methods*. The most
straight-forward approach, referred to as *policy iteration*, proceeds
in two steps: First, given a policy $\pi(a|s)$, the value function
$v_{\pi}(s)$ is evaluated. Second, after this *policy evaluation*, we
can improve on the given policy $\pi(a|s)$ using the greedy policy

```{math}
:label: eqn:greedy_improvement

\pi'(a|s) = {\rm argmax}_a  \sum_{s', r} p(s', r| s, a) [r + \gamma v_\pi(s')].
``` 

This second step is called *policy improvement*. The full policy iteration then proceeds iteratively

```{math}
\pi_0 \rightarrow v_{\pi_0} \rightarrow \pi_1 \rightarrow v_{\pi_1} \rightarrow \pi_2 \rightarrow \cdots
```

until convergence to $v_*$ and hence $\pi_*$. Note that, indeed, the Bellman optimality equation [](eqn:bellman-optimality-1) is the fixed-point equation for this procedure.

Policy iteration requires a full evaluation of the value function of
$\pi_k$ for every iteration $k$, which is usually a costly calculation.
Instead of fully evaluating the value function under a fixed policy, we
can also directly try and calculate the optimal value function by
iteratively solving the Bellman optimality equation,

```{math}
v^{[k+1]} (s) = \max_a \sum_{s', r} p(s', r| s, a) [r + \gamma v^{[k]}(s')].
```

Note that once we have converged to the optimal value function, the
optimal policy is given by the greedy policy corresponding to the
right-hand side of Eq. [](eqn:greedy_improvement). An alternative way of interpreting this iterative procedure is to perform policy improvement every time we update the value function, instead of finishing the policy evaluation each time before policy improvement. This procedure is called *value iteration* and is an example of a *generalized policy iteration*, the idea of allowing policy evaluation and policy improvement to interact while learning.

In the following, we want to use such a generalized policy iteration scheme for the (common) case, where we do not have a model for our environment. In this model-free case, we have to perform the (generalized) policy improvement using only our interactions with the environment. It is instructive to first think about how to evaluate a policy. We have seen in Eqs. [](eqn:value_function) and [](eqn:BE_expect) that the value function can also be written as an expectation value, 

```{math}
:label: eqn:policy_evaluation

\begin{aligned}
    v_{\pi} (s) &= E_\pi (G_t | S_t = s)\\
    &= E_\pi (R_{t+1} + \gamma v_{\pi}(S_{t+1})| S_t = s).
\end{aligned}
```

We can thus either try to directly sample the expectation value of the
first line---this can be done using *Monte Carlo sampling* over possible
state-action sequences---or we try to use the second line to iteratively
solve for the value function. In both cases, we start from state $S_t$
and choose an action $A_t$ according to the policy we want to evaluate.
The agent's interaction with the environment results in the reward
$R_{t+1}$ and the new state $S_{t+1}$. Using the second line,
Eq. [](eqn:policy_evaluation), goes under the name
*temporal-difference learning* and is in many cases the most efficient
method. In particular, we make the following updates

```{math}
:label: eqn:policy_evaluation_modelfree

v_\pi^{[k+1]} (S_t) = v_\pi^{[k]}(S_t) + \alpha [R_{t+1} + \gamma v_\pi^{[k]} (S_{t+1}) - v_\pi^{[k]} (S_{t}) ].
```

The expression in the brackets is the difference between our new estimate and the old estimate of the value function and $\alpha<1$ is a learning rate. As we look one step ahead for our new estimate, the method is called one-step temporal difference method.

We now want to use generalized policy iteration to find the optimal
value. We already encountered a major difficulty when improving a policy
using a value function based on experience in
Sec. [](sec:expl_v_expl): it is difficult to maintain enough
exploration over possible action-state pairs and not end up exploiting
the current knowledge. However, this sampling is crucial for both Monte
Carlo methods and the temporal-difference learning we discuss here. In
the following, we will discuss two different methods of performing the
updates, both working on the state-action value function, instead of the
value function. Both have in common that we look one step ahead to
update the state-action value function. A general update should then be
of the form

```{math}
q^{[k+1]} (S_t, a) = q^{[k]}(S_t, a) + \alpha [R_{t+1} + \gamma q^{[k]} (S_{t+1}, a') - q^{[k]} (S_{t}, a) ]
```

and the question is then what action $a$ we should take for the state-action pair and what action $a'$ should be taken in the new state $S_{t+1}$.

Starting from a state $S_0$, we first choose an action $A_0$ according
to a policy derived from the current estimate of the state-action value
function [^1], such as an $\epsilon$-greedy policy. For the first
approach, we perform updates as

```{math}
q^{[k+1]} (S_t, A_t) = q^{[k]}(S_t, A_t) + \alpha [R_{t+1} + \gamma q^{[k]} (S_{t+1}, A_{t+1}) - q^{[k]} (S_{t}, A_t) ].
```

As above, we are provided a reward $R_{t+1}$ and a new state $S_{t+1}$
through our interaction with the environment. To choose the action
$A_{t+1}$, we again use a policy derived from $Q^{[k]}(s=S_{t+1}, a)$.
Since we are using the policy for choosing the action in the next state
$S_{t+1}$, this approach is called *on-policy*. Further, since in this
particular case, we use the quintuple
$S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}$, this algorithm is referred to as
*Sarsa*. Finally, note that for the next step, we use $S_{t+1}, A_{t+1}$
as the state-action pair for which $q^{[k]}(s,a)$ is updated.

Alternatively, we only keep the state $S_t$ from the last step and first
choose the action $A_t$ for the update using the current policy. Then,
we choose our action from state $S_{t+1}$ in greedy fashion, which
effectively uses $Q^{[k]}(s=S_t, a)$ as an approximation for
$q_*(s=S_t, a)$. This leads to

```{math}
q^{[k+1]} (S_t, A_t) = q^{[k]}(S_t, A_t) + \alpha [R_{t+1} + \gamma \max_a q^{[k]} (S_{t+1}, a) - q^{[k]} (S_{t}, A_t) ].
```

and is a so-called *off-policy* approach. The algorithm, a variant of
which is used in AlphaGo, is called *Q-learning*.


[^1]: We assume here an episodic task. At the very beginning of
    training, we may initialize the state-action value function
    randomly.

