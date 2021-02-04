(sec:expl_v_expl)=
# Exploration versus Exploitation

We begin our discussion with a simple example that demonstrates some
important aspects of reinforcement learning. In particular, we discuss a
situation, where the reward does not depend on a state, but only on the
action taken. The agent is a doctor, who has to choose from $n$ actions,
the treatments, for a given disease, with the reward depending on the
recovery of the patient. The doctor 'learns on the job' and tries to
find the best treatment. The *value* of a treatment $a\in\mathcal{A}$ is
denoted by $q_* (a) = E( r )$, the expectation value of our reward.

Unfortunately, there is an uncertainty in the outcome of each treatment,
such that it is not enough to perform each treatment just once to know
the best one. Rather, only by performing a treatment many times we find
a good estimate $Q_t(a) \approx q_*(a)$. Here, $Q_t(a)$ is our estimate
of the value of $a$ after $t$ (time-) steps. Obviously, we should not
perform a bad treatment many times, only to have a better estimate for
its failure. We could instead try each action once and then continue for
the rest of the time with the action that performed best. This strategy
is called a *greedy* method and *exploits* our knowledge of the system.
Again, this strategy bears risks, as the uncertainty in the outcome of
the treatment means that we might use a suboptimal treatment. It is thus
crucial to *explore* other actions. This dilemma is called the 'conflict
between exploration and exploitation'. A common strategy is to use the
best known action $a_* = {\rm argmax}_a Q_t(a)$ most of the time, but
with probability $\epsilon$ chose randomly one of the other actions.
This strategy of choosing the next action is called *$\epsilon$-greedy*.


