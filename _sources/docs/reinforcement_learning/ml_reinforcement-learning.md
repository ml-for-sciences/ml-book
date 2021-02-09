<!-- Global site tag (gtag.js) - Google Analytics -->

<script async src="https://www.googletagmanager.com/gtag/js?id=G-ZLMLLKHZE0"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-ZLMLLKHZE0');
</script>
(sec:RL)=
# Reinforcement Learning

In the previous sections, we have introduced data-based learning, where
we are given a dataset $\{\mathbf{x}_i\}$ for training. Depending on whether
we are given labels $y_i$ with each data point, we have further divided
our learning task as either being supervised or unsupervised,
respectively. The aim of machine learning is then to classify unseen
data (supervised), or extract useful information from the data and
generate new data resembling the data in the given dataset
(unsupervised). However, the concept of learning as commonly understood
certainly encompasses other forms of learning that are not falling into
these data-driven categories.

An example for a form of learning not obviously covered by supervised or
unsupervised learning is learning how to walk: in particular, a child
that learns how to walk does not first collect data on all possible ways
of successfully walking to extract rules on how to walk best. Rather,
the child performs an action, sees what happens, and then adjusts their
actions accordingly. This kind of learning thus happens best
'on-the-fly', in other words while performing the attempted task.
*Reinforcement learning* formalizes this different kind of learning and
introduces suitable (computational) methods.

As we will explain in the following, the framework of reinforcement
learning considers an *agent*, that interacts with an *environment*
through actions, which, on the one hand, changes the *state* of the
agent and on the other hand, leads to a *reward*. Whereas we tried to
minimize a loss function in the previous sections, the main goal of
reinforcement learning is to maximize this reward by learning an
appropriate *policy*. One way of reformulating this task is to find a
*value function*, which associates to each state (or state-action pair)
a value, or expected total reward. Note that, importantly, to perform
our learning task we do not require knowledge, a *model*, of the
environment. All that is needed is feedback to our actions in the form
of a reward signal and a new state. We stress again that we study in the
following methods that learn at each time step. One could also devise
methods, where an agent tries a policy many times and judges only the
final outcome.

The framework of reinforcement learning is very powerful and versatile.
Examples include:

-   We can train a robot to perform a task, such as using an arm to
    collect samples. The state of the agent is the position of the robot
    arm, the actions move the arm, and the agent receives a reward for
    each sample collected.

-   We can use reinforcement learning to optimize experiments, such as
    chemical reactions. In this case, the state contains the
    experimental conditions, such as temperature, solvent composition,
    or pH and the actions are all possible ways of changing these state
    variables. The reward is a function of the yield, the purity, or the
    cost. Note that reinforcement learning can be used at several levels
    of this process: While one agent might be trained to target the
    experimental conditions directly, another agent could be trained to
    reach the target temperature by adjusting the current running
    through a heating element.

-   We can train an agent to play a game, with the state being the
    current state of the game and a reward is received once for winning.
    The most famous example for such an agent is Google's AlphaGo, which
    outperforms humans in the game of Go. A possible way of applying
    reinforcement learning in the sciences is to phrase a problem as a
    game. An example, where such rephrasing was successfully applied, is
    error correction for (topological) quantum computers.

-   In the following, we will use a toy example to illustrate the
    concepts introduced: We want to train an agent to help us with the
    plants in our lab: in particular, the state of the agent is the
    water level. The agent can turn on and off a growth lamp and it can
    send us a message if we need to show up to water the plants.
    Obviously, we would like to optimize the growth of the plants and
    not have them die.

As a full discussion of reinforcement learning goes well beyond the
scope of this lecture, we will focus in the following on the main ideas
and terminology with no claim of completeness.

