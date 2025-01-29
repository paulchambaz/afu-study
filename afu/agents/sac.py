import torch
import torch.nn as nn
import torch.nn.functional as F
from bbrl.agents import Agent, TemporalAgent  # type: ignore


class SquashedGaussianActor(Agent):
    """
    Implementation of SAC's policy network that outputs a squashed Gaussian
    distribution.

    In maximum entropy RL, the policy pi(a|s) is represented as a probability
    distribution over actions. Using a Gaussian allows us to sample continuous
    actions while maintaining differentiability through the reparametrizatin
    trick. The 'squashing' via tanh ensures actions remain bounded while
    preserving the policy's stochastic nature.
    """

    def __init__(self, state_dim, action_dim, hidden_size=256, min_std=1e-4):
        super().__init__()

        # The policy is modeled as a neural network that maps states to the
        # parameters of a Gaussian distribution (mean, std) for each action
        # dimension. The network uses two hidden layers to learn complex
        # state-action relationships. ReLU activations introduce non-linearity
        # while maintaining good gradient flow.
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Instead of directly outputting a fixed policy, we output a Gaussian
        # distribution for each action dimension. This requires two seperate
        # outputs. The mean represents the agent's best guess for optimal action.
        # The log_std represents exploration tendency, learned during training.
        self.mean_layer = nn.Linear(hidden_size, action_dim)
        self.log_std_layer = nn.Linear(hidden_size, action_dim)
        self.min_std = min_std

    def forward(self, t, stochastic=True):
        """
        Forward pass computes action distributions from states. The process
        follows:
        1. Extract state features through policy network
        2. Generate Gaussian parameters (mean, log_std)
        3. Sample actions using reprametrization (if stochastic)
        4. Apply squashing and compute log probabilities
        """
        obs = self.get(("env/env_obs", t))
        hidden = self.policy_net(obs)

        # The network outputs the mean directly but outputs log_std for numerical
        # stability. We apply softplus to ensure std > 0 and add min_std to
        # prevent the policy from becoming deterministic, which would halt
        # exploration.
        mean = self.mean_layer(hidden)
        log_std = self.log_std_layer(hidden)
        std = F.softplus(log_std) + self.min_std

        # Reparametrization trick: Instead of sampling directly from the gaussian,
        # we sample noise from a normal distribution and compute the action as a
        # random point from the initial gaussian prediction. This allows gradients
        # to flow through the sampling process, essential for policy optimization.
        if stochastic:
            noise = torch.randn_like(mean)
            raw_action = mean + std * noise
        else:
            raw_action = mean

        # The tanh squashing serves two purposes. First it bounds actions to
        # [-1, 1], making them suitable for environments and second it preserves
        # differentiability unlike hard clipping.
        action = torch.tanh(raw_action)

        # Log probability computation has two terms. First the standard Gaussian
        # log probability and second the squashing correction, which accounts for
        # the change in probability density due to tanh transformation
        # (derived from change of variables formula)
        if stochastic:
            log_prob = (-0.5 * (noise**2) - log_std).sum(-1)
            log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(-1)
            self.set(("action_logprobs", t), log_prob)

        self.set(("action", t), action)


class ContinuousQAgent(Agent):
    """
    Implementation of SAC's critic network that estimates action-values
    (Q-values).

    The critic learns to estimate expected future returns Q(s, a) for any
    state-action pair. Unlike traditional Q-learning, SAC's critic must handle
    continuous actions and incorporate entropy maximization. It serves as a
    learned reward model that helps guide the actor's policy optimization.
    """

    def __init__(self, state_dim, action_dim, hidden_size=256):
        super().__init__()

        # The Q-network takes concatenated state-action pairs as input and
        # outputs their estimated Q-values. Like the actor, it uses two hidden
        # layers to capture complex relationships. The final layer outputs
        # unbounded scalar values since Q-values have no inherent bounds.
        self.q_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        # Marker used by framework to identify this as a Q-function
        self.is_q_function = True

    def forward(self, t):
        """
        Forward pass estimates Q-value for a given state-action pair. The process
        follows:
        1. Get current observation and action from workspace
        2. Concatenate them to form the network input
        3. Compute and output estimated Q-value
        """
        obs = self.get(("env/env_obs", t))
        action = self.get(("action", t))
        state_action = torch.cat([obs, action], dim=-1)
        q_value = self.q_net(state_action).squeeze(-1)
        self.set((f"{self.prefix}q_value", t), q_value)


class SACAgent(Agent):
    """
    Implementation of the Soft Actor-Critic (SAC) algorithm.

    SAC uses multiple networks working together. First an actor that learns an
    optimal policy, then dual critics that prevent overestimation bias, finally
    target critics that provide stable learning targets. They all work together
    to learn a policy that maximizes both reward and entropy.
    """

    def __init__(self, state_dim, action_dim, cfg):
        super().__init__()

        # We create multiple neural networks, first the actor to learn the policy,
        # two critics to estimate Q-values and prevent overestimation and finally
        # two target critics that update slowly for training stability. Each
        # critic gets a unique prefix for workspace variable identification.
        self.actor = SquashedGaussianActor(state_dim, action_dim)
        self.critic1 = ContinuousQAgent(state_dim, action_dim).with_prefix(
            "critic_1/"
        )
        self.critic2 = ContinuousQAgent(state_dim, action_dim).with_prefix(
            "critic2/"
        )
        self.target_critic1 = ContinuousQAgent(state_dim, action_dim).with_prefix(
            "target1/"
        )
        self.target_critic2 = ContinuousQAgent(state_dim, action_dim).with_prefix(
            "target2/"
        )

        # Initialize target networks with copies of the original critics
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # Wrap core components in temporal agents to handle sequential decision
        # making. We maintain seperate actors for training and evaluation, and
        # group critics together using ModuleList for efficient processing.
        self.train_actor = TemporalAgent(self.actor)
        self.eval_actor = TemporalAgent(self.actor)
        self.critics = TemporalAgent(nn.ModuleList([self.critic1, self.critic2]))
        self.target_critics = TemporalAgent(
            nn.ModuleList([self.target_critic1, self.target_critic2])
        )

        # Core SAC hyperparameters:
        # tau controls speed of target network updates (smaller is more stable)
        self.tau = cfg.algorithm.tau_target

        # discount balances immediate vs future rewards
        self.discount = cfg.algorithm.discout_factor

        # target_entropy is desired policy randomness, it scales with action
        # space
        self.target_entropy = -action_dim

    def compute_critic_loss(
        cfg,
        reward,
        must_bootstrap,
        t_actor,
        t_q_agents,
        t_target_q_agents,
        rb_workspace,
        ent_coef,
    ):
        # Get next actions and their log probs from current policy
        t_actor(rb_workspace, t=1, n_steps=1, stochastic=True)

        # Get Q-values from target critics for next state-action
        t_target_q_agents(rb_workspace, t=1)

        # Get the min of two target Q-values
        target_q1 = rb_workspace["target-critic-1/q_value"][1].detach()
        target_q2 = rb_workspace["target-critic-2/q_value"][1].detach()
        min_target_q = torch.min(target_q1, target_q2)

        # Add entropy term
        action_logprobs = rb_workspace["action_logprobs"][1].detach()
        min_target_q = min_target_q - ent_coef * action_logprobs

        # Compute target using reward and discounted next q-value
        target = (
            reward + cfg.algorithm.discount_factor * must_bootstrap * min_target_q
        )

        # Get current Q-values
        t_q_agents(rb_workspace, t=0)
        q1 = rb_workspace["critic-1/q_value"][0]
        q2 = rb_workspace["critic-2/q_value"][0]

        # Compute MSE loss for both critics
        critic_loss_1 = F.mse_loss(q1, target)
        critic_loss_2 = F.mse_loss(q2, target)

        return critic_loss_1, critic_loss_2

    def compute_actor_loss(ent_coef, t_actor, t_q_agents, rb_workspace):
        # Get actions and log probs from current policy
        t_actor(rb_workspace, t=0, n_steps=1, stochastic=True)

        # Get Q-values for current actions
        t_q_agents(rb_workspace, t=0)
        q1 = rb_workspace["critic-1/q_value"][0]
        q2 = rb_workspace["critic-2/q_value"][0]
        min_q = torch.min(q1, q2)

        # Get log probs
        log_probs = rb_workspace["action_logprobs"][0]

        # Compute actor loss: negative Q-value + entropy term
        actor_loss = (ent_coef * log_probs - min_q).mean()

        return actor_loss


def soft_update(target, source, tau):
    """
    Gradually update target network parameters towards source network. Use
    polyak averaging, which creates a slowly-moving average of the source
    network, providing stability during training.
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )
