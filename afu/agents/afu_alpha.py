import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import bbrl
from bbrl.workspace import Workspace
from bbrl.agents import Agent, TemporalAgent
from .memory import ReplayBuffer

# Définition de la fonction de valeur V(s) sous forme d'Agent BBRL
# Cette fonction estime la valeur d'un état s, utilisée pour calculer la valeur cible
class ValueFunction(Agent):
    def __init__(self, state_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, workspace: Workspace, t: int):
        state = workspace.get(("state", t))
        value = self.model(state)
        workspace.set(("value", t), value)

# Définition de la classe Critic qui regroupe deux estimateurs Q
class Critic(Agent):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, workspace: Workspace, t: int):
        state = workspace.get(("state", t))
        action = workspace.get(("action", t))
        state_action = torch.cat([state, action], dim=-1)
        q1_value = self.q1(state_action)
        q2_value = self.q2(state_action)
        workspace.set(("q1_value", t), q1_value)
        workspace.set(("q2_value", t), q2_value)

# Définition de l'acteur sous forme d'Agent BBRL
class Actor(Agent):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
    
    def forward(self, workspace: Workspace, t: int):
        state = workspace.get(("state", t))
        action = self.model(state)
        workspace.set(("action", t), action)

# Définition de l'algorithme AFU-alpha avec BBRL
class AFU:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, buffer_size=10000, batch_size=64):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.value = ValueFunction(state_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.actor = Actor(state_dim, action_dim).to(self.device)
        
        self.critic_agent = TemporalAgent(self.critic)
        self.actor_agent = TemporalAgent(self.actor)
        
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        self.gamma = gamma
        self.batch_size = batch_size
    
    def select_action(self, state):
        """Sélectionne une action en fonction d'un état donné."""
        workspace = Workspace()
        workspace.set(("state", 0), torch.tensor(state, dtype=torch.float32).unsqueeze(0))
        self.actor_agent(workspace, t=0, n_steps=1)
        return workspace.get(("action", 0)).squeeze(0).detach().numpy()
    
    # Sample action a ∼ πθ (·|s).
    # Perform environment step s, a → s′ , compute r = R(s, a), and insert (s, a, r, s′ ) in Rb.
    def update(self):
        # for each gradient step do
        # Draw batch of transitions B from Rb and compute loss gradients on that batch.
        # ψ ← ψ − ηQ ∇ψ LQ (ψ)
        # φi∈{1,2} ← φi − ηV,A ∇φi LV,A (φi , ξi )
        # ξi∈{1,2} ← ξi − ηV,A ∇ξi LV,A (φi , ξi )
        # φtarget i∈{1,2} ← τ φi + (1 − τ )φtarget i
        # θ ← θ − ηπ ∇θ Lπ (θ)
        # α ← α − ηtemp ∇α Ltemp(α)
        # end for
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0
        
        workspace = Workspace()
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size, continuous=True)
        
        workspace.set(("state", 0), states)
        workspace.set(("action", 0), actions)
        workspace.set(("reward", 0), rewards)
        workspace.set(("next_state", 0), next_states)
        workspace.set(("done", 0), dones)
        
        self.critic_agent(workspace, t=0, n_steps=1)
        self.actor_agent(workspace, t=0, n_steps=1)
        
        q1_value, q2_value = workspace.get(("q1_value", 0)), workspace.get(("q2_value", 0))
        min_q = torch.min(q1_value, q2_value)
        target = rewards + self.gamma * min_q * (1 - dones)
        
        critic_loss = nn.MSELoss()(min_q, target.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        actor_loss = -q1_value.mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return actor_loss.item(), critic_loss.item()
    
    def save(self, path: str):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict()
        }, path)
    
    def load(self, path: str):
        save_dict = torch.load(path)

        self.policy.load_state_dict(save_dict["policy_state"])
        self.value.load_state_dict(save_dict["value_state"])
        self.value_target.load_state_dict(save_dict["value_target_state"])
        self.q1.load_state_dict(save_dict["q1_state"])
        self.q2.load_state_dict(save_dict["q2_state"])

        self.policy_optimizer.load_state_dict(save_dict["policy_optimizer_state"])
        self.value_optimizer.load_state_dict(save_dict["value_optimizer_state"])
        self.q1_optimizer.load_state_dict(save_dict["q1_optimizer_state"])
        self.q2_optimizer.load_state_dict(save_dict["q2_optimizer_state"])

        self.params = save_dict["params"]
        self.total_steps = save_dict["total_steps"]
    
    @classmethod
    def load_agent(cls, path: str):
        save_dict = torch.load(path)
        agent = cls(save_dict["params"])
        agent.load(path)
        return agent
