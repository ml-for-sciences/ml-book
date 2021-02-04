
## Policies and Value Functions

A policy $\pi(a | s)$ is the probability of choosing the action $a$ when
in state $s$. We can thus formulate our learning task as finding the
policy that maximizes our reward and reinforcement learning as adapting
an agent's policy as a result of its experience. For a given policy, we
can define the value function of a state $s$ as the expected return from
starting in that state and using the policy function $\pi$ for choosing
all our future actions. We can write this as

```{math}
:label: eqn:value_function

v_\pi(s) \equiv E_\pi (G_t | S_t = s).
```

Alternatively, we can define the action-value function of $\pi$ as

```{math}
q_\pi (s, a) \equiv E_\pi(G_t | S_t = s, A_t = a).
```

This is the expectation value for the return starting in state $s$ and choosing
action $a$, but using the policy $\pi$ for all future actions. Note that
one of the key ideas of reinforcement learning is to use such value
functions, instead of the policy, to organize our learning process.

The value function of Eq. [](eqn:value_function) satisfies a self-consistency equation,

```{math}
:label: eqn:BE_expect
\begin{aligned}
    v_\pi(s) &= E_\pi ( R_{t+1} + \gamma G_{t+1} | S_t = s)\\
    &= \sum_a \pi(a | s) \sum_{s', r} p(s', r | s, a) [ r + \gamma v_\pi (s')].\end{aligned}
```

This equation, known as the *Bellman equation*, relates the value of
state $s$ to the expected reward and the (discounted) value of the next
state after having chosen an action under the policy $\pi(a|s)$.

As an example, we can write the Bellman equation for the strategy of
always leaving the lamps on in our toy model. Then, we find the system
of linear equations 

```{math}
\begin{aligned}
    v_{\rm on}({\rm h}) &= p({\rm h}, r_{\rm on} | {\rm h}, {\rm on}) [r_{\rm on} + \gamma v_{\rm on} ({\rm h})] + p({\rm l}, r_{\rm on} | {\rm h}, {\rm on}) [r_{\rm on} + \gamma v_{\rm on} ({\rm l})] \nonumber\\
    & = r_{\rm on} + \gamma [\alpha v_{\rm on}({\rm h}) + (1-\alpha) v_{\rm on}({\rm l})],\\
    v_{\rm on}({\rm l}) &=  \beta[r_{\rm on} + \gamma v_{\rm on}({\rm l})] + (1-\beta) [r_{\rm fail} + \gamma v_{\rm on}({\rm h})],
\end{aligned}
```

from which we can solve easily for $v_{\rm on}({\rm h})$ and $v_{\rm on}({\rm l})$.

Instead of calculating the value function for all possible policies, we
can directly try and find the optimal policy $\pi_*$, for which
$v_{\pi_*}(s) > v_{\pi'}(s)$ for all policies $\pi'$ and
$s\in\mathcal{S}$. For this policy, we find the *Bellman optimality
equations* 

```{math}
:label: eqn:bellman-optimality-1
\begin{aligned}
    v_*(s) &= \max_a q_{\pi_*}(s, a)\nonumber\\
    &= \max_a E(R_{t+1} + \gamma v_*(S_{t+1}) | S_t = s, A_t = a)\\
    &=\max_a \sum_{s', r} p(s', r | s, a) [ r + \gamma v_* (s')].
\end{aligned}
```

Importantly, the Bellman optimality equations do not depend on the actual policy anymore. As such, Eq. [](eqn:bellman-optimality-1) defines a non-linear system of equations, which for a sufficiently simple MDP can be solved explicitly.
For our toy example, the two equations for the value functions are

```{math}
v_*({\rm h}) = \max \left\{\begin{array}{l} r_{\rm on} + \gamma [\alpha v_*({\rm h}) + (1-\alpha) v_*({\rm l})] \\ 
    r_{\rm off} + \gamma [\alpha' v_*({\rm h}) + (1-\alpha') v_*({\rm l})] \end{array}\right.
```

and

```{math}
v_*({\rm l}) = \max \left\{\begin{array}{l} \beta[r_{\rm on} + \gamma v_*({\rm l})] + (1-\beta) [r_{\rm fail} + \gamma v_*({\rm h})] \\ 
    \beta'[r_{\rm off} + \gamma v_*({\rm l})] + (1-\beta') [r_{\rm fail} + \gamma v_*({\rm h})]\\
    r_{\rm text} + \gamma v_*({\rm h})\end{array}\right. .
```

Note that equivalent equations to Eqs. [](eqn:bellman-optimality-1) hold for the state-action value function 

```{math}
:label: eqn:bellman-optimality-2
\begin{aligned}
    q_*(s, a) &= E(R_{t+1} + \gamma \max_{a'} q_*(S_{t+1},a'))\\
    &= \sum_{s', r} p(s', r | s, a) [ r + \gamma \max_{a'} q_* (s', a')].
\end{aligned}
```

Once we know $v_*$, the optimal policy $\pi_* (a| s)$ is the greedy
policy that chooses the action $a$ that maximizes the right-hand side of
Eq. [](eqn:bellman-optimality-1). If, instead, we know $q_*(s,a)$,
then we can directly choose the action which maximizes $q_*(s,a)$,
namely $\pi_*(a | s) = {\rm argmax}_{a'} q_*(s, a')$, without looking
one step ahead.

While Eqs [](eqn:bellman-optimality-1) or [](eqn:bellman-optimality-2) can be solved explicitly for a sufficiently simple system, such an approach, which corresponds to an exhaustive search, is often not feasible. In the following, we distinguish two levels of complexity: First, if the explicit solution is too hard, but we can still keep track of all possible value functions---we can choose either the state or the state-action value function---we can use a *tabular* approach. A main difficulty in this case is the evaluation of a policy, or prediction, which is needed to improve on the policy. While various methods for *policy evaluation* and *policy improvement* exist, we will discuss in the following an approach called *temporal-difference learning*. Second, in many cases the space of possible states is much too large to allow for a complete knowledge of all value functions. In this case, we additionally need to approximate the value functions. For this purpose, we can use the methods encountered in the previous chapters, such as (deep) neural networks.




