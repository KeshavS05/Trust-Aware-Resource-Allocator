import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Tuple, Dict, Any, Optional
import networkx as nx
import torch.nn.utils as nn_utils
import os



# ================================================================================================================================================
# ================================================================================================================================================


import json

def load_graph_jsonl(path: str):
    samples = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples

def build_nx_graph_from_edges(n: int, edges, undirected=True) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from((int(u), int(v)) for u, v in edges)
    # If edges are directed in file but you want undirected, nx.Graph already treats as undirected
    return G

class GraphSampler:
    """
    Cycles or randomly samples graphs from a JSONL file.
    Each sample dict is expected to contain at least: n, edges, degree, family, graph_id.
    """
    def __init__(self, jsonl_path: str, seed: int = 0, shuffle: bool = True):
        self.samples = load_graph_jsonl(jsonl_path)
        assert len(self.samples) > 0, "No samples loaded from JSONL."
        self.rng = np.random.default_rng(seed)
        self.shuffle = shuffle
        self.order = np.arange(len(self.samples))
        if shuffle:
            self.rng.shuffle(self.order)
        self.ptr = 0

    def sample(self):
        idx = int(self.order[self.ptr])
        self.ptr = (self.ptr + 1) % len(self.samples)
        if self.ptr == 0 and self.shuffle:
            self.rng.shuffle(self.order)

        s = self.samples[idx]
        n = int(s["n"])
        G = build_nx_graph_from_edges(n, s["edges"], undirected=True)

        # degree vec: use provided degrees if present; else compute from G
        if "degree" in s and s["degree"] is not None:
            degs = np.array(s["degree"], dtype=np.float32)
        else:
            degs = np.array([G.degree(i) for i in range(n)], dtype=np.float32)

        degs_norm = degs.copy()
        if degs_norm.max() > 0:
            degs_norm = degs_norm / degs_norm.max()

        meta = {
            "graph_id": s.get("graph_id", None),
            "family": s.get("family", None),
            "params": s.get("params", None),
            "seed": s.get("seed", None),
        }
        return G, degs_norm, meta




def project_to_simplex(x, eps=1e-8):
   """
   Project vector into 0 to 1
   """
   x = np.maximum(x, eps)
   return x / x.sum()


class AllocationEnvStage1:
   """
   Given wants w try to allocate resources optimally across different nodes
   to minimize shortfall penalty (w_i - a_i)^2. Using one step episode here
      - Observation: [wants, degree_dummy] (size 2N)
      - Action: allocation vector (size N), projected to simplex
      - Reward: - lambda_short * sum((w_i - a_i)^2)
      
   Parameters:
      - n_nodes: number of nodes
      - max_want: maximum want value per node (i.e how much is the max a node may request)
      - lambda_short: penalty weight for shortfall (larger means more emphasis on minimizing shortfall)
      - seed: random seed for reproducibility
      - wants_init: optional fixed wants vector for testing
   """

   def __init__(self, n_nodes: int, max_want: float = 1.0, lambda_short: float = 5.0, seed: Optional[int] = None, wants_init: Optional[np.ndarray] = None,):
      
      self.n = n_nodes
      self.max_want = max_want
      self.lambda_short = lambda_short
      self.rng = np.random.default_rng(seed)
      self.state: Optional[np.ndarray] = None
      self.degree_dummy = np.zeros(self.n, dtype=np.float32)
      self.trust_dummy = np.zeros(self.n, dtype=np.float32)
      self.wants_init = wants_init

   def _sample_wants(self):
      '''
      Sample wants vector for current episode. If wants_init is provided, use that instead.
      '''
      if self.wants_init is not None:                                                     # use fixed wants for testing
         return self.wants_init.copy()
      return self.rng.uniform(0.0, self.max_want, size=self.n)                            # sample wants uniformly

   def reset(self):
      '''
      Reset environment state by sampling new wants vector. Concatenate with degree dummy,
      so the observation size is 2n. Return the initial observation.
      '''
      wants = self._sample_wants()                                                        # sample wants (n,)
      obs = np.concatenate([wants, self.trust_dummy, self.degree_dummy])                                    # observation (2n,)
      self.state = obs                                                                    # set initial state
      return obs.copy()           

   def step(self, action):
      '''
      Take an action (allocation vector), compute shortfall penalty and reward,
      sample new wants for next state, and return (next_obs, reward, done, info).
      '''
      assert self.state is not None, "Call reset() before step()."
      assert action.shape == (self.n,)

      w = self.state[:self.n]                                                             # current wants (n,)
      alloc = project_to_simplex(action)                                                  # projected allocation (n,)

      # shortfall = np.maximum(w - alloc, 0.0)
      shortfall = np.maximum(w - alloc, 0.0)                                                       # absolute shortfall (n,)
      oversupply = np.maximum(alloc - w, 0.0)  
      lambda_over = 2 * self.lambda_short
      penalty_short = np.sum(self.lambda_short * shortfall**2 + lambda_over * oversupply**2)                                              # shortfall penalty (scalar)

      reward = -1 * penalty_short                                    # reward is negative penalty
      done = True                                                                         # single-step episode

      next_wants = self._sample_wants()                                                   # sample new wants for next state (n,)
      next_obs = np.concatenate([next_wants, self.trust_dummy, self.degree_dummy])                          # next observation (2n,)

      info = {                                                                            # extra info for debugging
         "wants": w,
         "alloc": alloc,
         "shortfall": shortfall,
         "penalty_short": penalty_short,
      }

      self.state = next_obs                                                               # update state
      return next_obs.copy(), reward, done, info                                          # return values



# ================================================================================================================================================
# ================================================================================================================================================



class ResourceAllocTrustEnv:
   """
   This environment models the allocation of resources among nodes in a network, considering both the nodes' 
   wants AND their trust levels. The trust dynamics are influenced by local satisfaction and neighbor interactions, 
   and the network structure is defined by a graph.

   Key features:
   - Observation: Concatenation of wants and normalized degree vector.
   - Action: Allocation vector, projected to a simplex.
   - Reward: Combines trust improvement and shortfall penalty.
   - Trust dynamics: Updated based on satisfaction and neighbor influence.
   - Network structure: Supports different graph types (e.g., star, line, two clusters).

   Parameters:
   - n_nodes: Number of nodes in the network.
   - episode_length: Number of steps in an episode.
   - alpha: Controls the rate of wants update.
   - trust_alpha, trust_beta: Parameters for the Beta distribution of initial trust.
   - trust_eta: Learning rate for trust updates.
   - lambda_short: Weight for shortfall penalty in the reward.
   - p: Exponent for satisfaction calculation.
   - neighbor_weight: Weight for neighbor influence on trust.
   - graph_type: Type of graph structure (e.g., "star", "line").
   - graph: Optional custom graph.
   """

   def __init__(
      self,
      n_nodes,
      episode_length=10,
      alpha=0.2,
      seed: Optional[int] = None,
      max_want=1.0,
      trust_alpha=2.0,
      trust_beta=5.0,
      trust_eta=0.5,
      lambda_short=2.0,
      p=1.3,
      neighbor_weight=0.35,
      wants_init: Optional[np.ndarray] = None,
      trust_init: Optional[np.ndarray] = None,
      graph_type="core_periphery",
      graph: Optional[nx.Graph] = None,
      graph_sampler=None,
      resample_graph_each_reset=True,
   ):
      
      self.n = n_nodes
      self.episode_length = episode_length
      self.alpha = alpha
      self.max_want = max_want

      self.trust_alpha = trust_alpha
      self.trust_beta = trust_beta
      self.trust_eta = trust_eta
      self.lambda_short = lambda_short
      self.p = p

      self.rng = np.random.default_rng(seed)
      self.state: Optional[np.ndarray] = None
      self.trust: Optional[np.ndarray] = None
      self.t = 0
      self.neighbor_weight = neighbor_weight
      self.wants_init = wants_init
      self.trust_init = trust_init
      
      
      
      
      self.graph_sampler = graph_sampler
      self.resample_graph_each_reset = resample_graph_each_reset

      if graph is not None:
         assert graph.number_of_nodes() == n_nodes, "graph and n_nodes mismatch"
         self.graph = graph
         degs = np.array([self.graph.degree(i) for i in range(self.n)], dtype=np.float32)
         if degs.max() > 0:
            degs = degs / degs.max()
         self.degree_vec = degs
         self.graph_meta = {}
         
      elif self.graph_sampler is not None:
         G, degs_norm, meta = self.graph_sampler.sample()
         assert G.number_of_nodes() == n_nodes, "sampled graph and n_nodes mismatch"
         self.graph = G
         self.degree_vec = degs_norm.astype(np.float32)
         self.graph_meta = meta
         
      else:
         raise ValueError("Either graph or graph_sampler must be provided.")



   def _sample_initial_wants(self):
      '''
      Sample initial wants vector for the episode. If wants_init is provided, use that instead.
      '''
      if self.wants_init is None:                                                         # sample wants uniformly
         return self.rng.uniform(0.0, self.max_want, size=self.n)
      else:
         return self.wants_init.copy()                                                   # use fixed wants for testing


   def _sample_initial_trust(self) -> np.ndarray:
      '''
      Sample initial trust vector for the episode.
      '''
      
      if self.trust_init is not None:
         return self.trust_init.copy()                                                   # use fixed trust for testing
      return self.rng.beta(self.trust_alpha, self.trust_beta, size=self.n)                # sample trust from Beta distribution


   def reset(self):
      
      if self.graph_sampler is not None and self.resample_graph_each_reset:
         G, degs_norm, meta = self.graph_sampler.sample()
         assert G.number_of_nodes() == self.n
         self.graph = G
         self.degree_vec = degs_norm.astype(np.float32)
         self.graph_meta = meta
         
      
      self.state = self._sample_initial_wants()                                           # initial wants
      self.trust = self._sample_initial_trust()                                           # initial trust
      self.t = 0

      for i in range(self.n):                                                             # set node attributes for wants and trust
         self.graph.nodes[i]["want"] = float(self.state[i])
         self.graph.nodes[i]["trust"] = float(self.trust[i])

      obs = np.concatenate([self.state, self.trust, self.degree_vec], axis=0)                         # initial observation
      return obs.copy()


   def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
      '''
      Take an action (allocation vector), update wants and trust, compute reward,
      and return (next_obs, reward, done, info).
      '''
      
      assert self.state is not None, "Call reset() before step()."
      assert self.trust is not None, "Trust not initialized; call reset()."
      assert action.shape == (self.n,)

      alloc = project_to_simplex(action)
      wants_t = self.state
      trust_t = self.trust

      oversupply = np.maximum(alloc - wants_t, 0.0)                                       # Oversupply calulated by difference between allocation and wants
      shortfall_trust = np.maximum(wants_t - alloc, 0.0)                                  # Shortfall adjusted by trust levels

      # frac = shortfall_trust / (wants_t + 1e-8)                                                 # Fraction of shortfall relative to wants
      # satisf = np.clip(frac, 0.0, 1.0)                                                    # Satisfaction is 1 - fraction shortfall (clipped to [0,1])
      served_frac = np.clip(np.minimum(alloc, wants_t) / (wants_t + 1e-8), 0.0, 1.0)
      satisf = np.power(served_frac, self.p)                                            # Non-linear scaling of satisfaction

      trust_local = trust_t + self.trust_eta * (satisf - trust_t)                         # Represents local trust update based on satisfaction
      trust_local = np.clip(trust_local, 0.0, 1.0)                                        # Clip trust to [0,1]

      neigh_mean = np.zeros_like(trust_t)                                                 # Mean trust of neighbors
      for i in range(self.n):
         neighbors = list(self.graph.neighbors(i))
         if len(neighbors) == 0:                                                         # No neighbors case
               neigh_mean[i] = trust_t[i]
         else:
               neigh_mean[i] = float(np.mean(trust_local[neighbors]))

      w = self.neighbor_weight        
      trust_next = (1.0 - w) * trust_local + w * neigh_mean                               # Combine local trust and neighbor influence
      trust_next = np.clip(trust_next, 0.0, 1.0)                                          # Clip trust to [0,1]

      mean_trust_t = float(np.mean(trust_t))                                              # Mean trust at current time
      mean_trust_next = float(np.mean(trust_next))                                        # Mean trust at next time
      # trust_gain = float(np.sum(trust_next - trust_t)) / self.n
      
      lambda_over = 2 * self.lambda_short
      penalty_short = np.sum(self.lambda_short * shortfall_trust**2 + lambda_over * oversupply**2)

      # reward = (mean_trust_next - mean_trust_t) - (penalty_short / self.n)    # Reward combines trust improvement and shortfall penalty
      weights = (self.degree_vec + 1e-6)
      weights = weights / weights.sum()
      trust_gain = np.sum(weights * (trust_next - trust_t))
      reward = 8 * trust_gain - 1.0 * (penalty_short / self.n)
      # reward = 1.2 * trust_gain - 0.8 * (penalty_short / self.n)
      
      wants_next = shortfall_trust + self.alpha * (served_frac)                             # Update wants based on shortfall and alpha
      wants_next = np.clip(wants_next, 0.0, self.max_want)                                           

      self.state = wants_next
      self.trust = trust_next
      self.t += 1

      for i in range(self.n):
         self.graph.nodes[i]["want"] = float(wants_next[i])
         self.graph.nodes[i]["trust"] = float(trust_next[i])

      done = (self.t >= self.episode_length)                                              # Episode termination condition
      obs_next = np.concatenate([wants_next, self.trust, self.degree_vec], axis=0)                    # Next observation

      info: Dict[str, Any] = {
         "wants_t": wants_t,
         "alloc": alloc,
         "shortfall": shortfall_trust,
         "oversupply": oversupply,
         "satisfaction": satisf,
         "trust_t": trust_t,
         "trust_next": trust_next,
         "mean_trust_t": mean_trust_t,
         "mean_trust_next": mean_trust_next,
         "penalty_short": penalty_short,
         "t": self.t,
         "graph_id": self.graph_meta.get("graph_id", None),
         "family": self.graph_meta.get("family", None),
      }
      return obs_next.copy(), reward, done, info



# ================================================================================================================================================
# ================================================================================================================================================



class ReplayBuffer:
   '''
   A simple FIFO experience replay buffer for DDPG.
   '''
   def __init__(self, capacity):
      self.buffer = deque(maxlen=capacity)                                                # buffer to store transitions

   def push(self, state, action, reward, next_state, done):
      '''
      Store a transition in the buffer.
      '''
      self.buffer.append((                                                                # store transition as a tuple
         np.array(state, copy=False),
         np.array(action, copy=False),
         float(reward),
         np.array(next_state, copy=False),
         float(done),
      ))

   def sample(self, batch_size):
      '''
      Sample a batch of transitions from the buffer.
      '''
      batch = random.sample(self.buffer, batch_size)
      states, actions, rewards, next_states, dones = map(np.array, zip(*batch))           # unzip and convert to numpy arrays
      return (                                                                            # return as torch tensors
         torch.from_numpy(states).float(),
         torch.from_numpy(actions).float(),
         torch.from_numpy(rewards).float().view(-1),
         torch.from_numpy(next_states).float(),
         torch.from_numpy(dones).float().view(-1),
      )

   def __len__(self):
      return len(self.buffer)



# ================================================================================================================================================
# ================================================================================================================================================



class Actor(nn.Module):
   '''
   Actor network: maps state to allocation vector (action).
   Output is passed through softmax to ensure allocations sum to 1.
   
   Parameters:
      - state_dim: dimension of the state input
      - n: number of nodes (size of allocation vector)
      - hidden_dim: size of hidden layers
      - temperature: softmax temperature for exploration control
   '''

   def __init__(self, state_dim, n, hidden_dim=128, temperature=1.5):
      '''
      Initialize the Actor network.
      '''
      super().__init__()
      self.temperature = temperature
      
      # Architecture: state_dim -> hidden_dim -> hidden_dim -> n
      self.net = nn.Sequential(            
         nn.Linear(state_dim, hidden_dim),                                               
         nn.ReLU(),
         nn.Linear(hidden_dim, hidden_dim),
         nn.ReLU(),
         nn.Linear(hidden_dim, n),
      )
      
      nn.init.uniform_(self.net[-1].weight, -1e-3, 1e-3)                                  # small init for final layer
      nn.init.uniform_(self.net[-1].bias, -1e-3, 1e-3)

   def forward(self, state: torch.Tensor) -> torch.Tensor:
      if state.dim() == 1:                                                                # single state case
         state = state.unsqueeze(0)                                                      # add batch dimension
      logits = self.net(state)
      alloc = torch.softmax(logits / self.temperature, dim=-1)                            # softmax to get allocation vector
      return alloc



# ================================================================================================================================================
# ================================================================================================================================================



class Critic(nn.Module):
   '''
   Critic network: maps (state, action) pair to Q-value.
   
   Parameters:
      - state_dim: dimension of the state input
      - n: number of nodes (size of action vector)
      - hidden_dim: size of hidden layers
   '''

   def __init__(self, state_dim, n, hidden_dim=128):
      super().__init__()
      
      # Architecture: (state_dim + n) -> hidden_dim -> hidden_dim -> 1
      self.net = nn.Sequential(
         nn.Linear(state_dim + n, hidden_dim),
         nn.ReLU(),
         nn.Linear(hidden_dim, hidden_dim),
         nn.ReLU(),
         nn.Linear(hidden_dim, 1),
      )

   def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
      if state.dim() == 1:                
         state = state.unsqueeze(0)
      if action.dim() == 1:
         action = action.unsqueeze(0)
      x = torch.cat([state, action], dim=-1)                                              # concatenate state and action
      return self.net(x).squeeze(-1)                                                      # return Q-value (squeeze last dim)




# ================================================================================================================================================
# ================================================================================================================================================



class DDPGAgent:
   '''
   DDPG agent for learning resource allocation policies.
   Implements actor-critic architecture with experience replay and target networks.
   
   Parameters:
      - state_dim: dimension of the state input
      - n_nodes: number of nodes (size of action vector)
      - gamma: discount factor
      - actor_lr: learning rate for actor network
      - critic_lr: learning rate for critic network
      - tau: soft update rate for target networks
      - buffer_capacity: capacity of the replay buffer
      - batch_size: minibatch size for training
      - device: computation device ("cpu" or "cuda")
   '''
   
   def __init__(
      self,
      state_dim,
      n_nodes,
      gamma=0.9,
      actor_lr=1e-4,
      critic_lr=1e-3,
      tau=0.001,
      buffer_capacity=100_000,
      batch_size=64,
      device: Optional[str] = None,
   ):
      self.n = n_nodes
      self.gamma = gamma
      self.tau = tau
      self.batch_size = batch_size
      self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

      self.actor = Actor(state_dim, n_nodes).to(self.device)
      self.actor_target = Actor(state_dim, n_nodes).to(self.device)
      self.actor_target.load_state_dict(self.actor.state_dict())

      self.critic = Critic(state_dim, n_nodes).to(self.device)
      self.critic_target = Critic(state_dim, n_nodes).to(self.device)
      self.critic_target.load_state_dict(self.critic.state_dict())

      self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
      self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

      self.replay_buffer = ReplayBuffer(buffer_capacity)
      self.mse_loss = nn.MSELoss()

   
   def save(self, path: str, extra: Optional[dict] = None):
         os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
         ckpt = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "gamma": self.gamma,
            "tau": self.tau,
            "n": self.n,
            "extra": extra or {},
         }
         torch.save(ckpt, path)

   def load(self, path: str, map_location: Optional[str] = None, load_optim: bool = True):
      loc = map_location or self.device
      ckpt = torch.load(path, map_location=loc)

      self.actor.load_state_dict(ckpt["actor"])
      self.critic.load_state_dict(ckpt["critic"])
      self.actor_target.load_state_dict(ckpt["actor_target"])
      self.critic_target.load_state_dict(ckpt["critic_target"])

      if load_optim and "actor_opt" in ckpt and "critic_opt" in ckpt:
         self.actor_opt.load_state_dict(ckpt["actor_opt"])
         self.critic_opt.load_state_dict(ckpt["critic_opt"])

      return ckpt.get("extra", {})
   
   
   def load_actor_weights(self, state_dict):
      '''
      Load pretrained weights into actor and its target network after Stage 1 training.
      '''
      self.actor.load_state_dict(state_dict)
      self.actor_target.load_state_dict(state_dict)

   def select_action(self, state, noise_scale=0.1):
      '''
      Select action given state, with optional Gaussian noise for exploration.
      '''
      self.actor.eval()                                                                   # set actor to eval mode              
      with torch.no_grad():                                                   
         s = torch.from_numpy(state).float().to(self.device)                             # convert state to tensor
         alloc = self.actor(s)                                                           # get allocation from actor
         
      self.actor.train()                                                                  # set actor back to train mode
      alloc = alloc.cpu().numpy().squeeze(0)                                              # convert allocation to numpy array

      if noise_scale > 0.0:                                                               # add Gaussian noise for exploration
         noise = np.random.normal(0.0, noise_scale, size=alloc.shape)                    # sample noise
         alloc = alloc + noise                                                           # add noise
         alloc = project_to_simplex(alloc)                                               # re-project to [0,1]

      return alloc

   def push_transition(self, state, action, reward, next_state, done): 
      '''
      Store a transition in the replay buffer.
      '''                    
      self.replay_buffer.push(state, action, reward, next_state, done)

   def _soft_update(self, net: nn.Module, target_net: nn.Module):
      '''
      Soft update target network parameters.
      '''
      for param, target_param in zip(net.parameters(), target_net.parameters()):
         target_param.data.copy_(
               self.tau * param.data + (1.0 - self.tau) * target_param.data                # soft update rule
         )

   def train_step(self):
      '''
      Perform a single training step for actor and critic networks.
      Return actor and critic losses.
      '''
      if len(self.replay_buffer) < self.batch_size:                                                   # not enough samples to train
         return None, None

      states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)       # sample minibatch
      states = states.to(self.device)
      actions = actions.to(self.device)
      rewards = rewards.to(self.device)
      next_states = next_states.to(self.device)
      dones = dones.to(self.device)

      with torch.no_grad():                                                                           # Critic update
         next_actions = self.actor_target(next_states)                                               # target actions from target actor
         target_q = self.critic_target(next_states, next_actions)                                    # target Q-values from target critic
         y = rewards + self.gamma * (1.0 - dones) * target_q                                         # TD target

      q_values = self.critic(states, actions)                                                         # current Q-values from critic
      critic_loss = self.mse_loss(q_values, y)                                                        # critic loss

      self.critic_opt.zero_grad()
      critic_loss.backward()
      nn_utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)                                # gradient clipping
      self.critic_opt.step()

      pred_actions = self.actor(states)                                                               # predicted actions from actor
      actor_loss = -self.critic(states, pred_actions).mean()                                          # actor loss (maximize Q-value)

      self.actor_opt.zero_grad()                                                                      
      actor_loss.backward()
      nn_utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)                                 # gradient clipping
      self.actor_opt.step()

      self._soft_update(self.actor, self.actor_target)                                                # soft update actor target
      self._soft_update(self.critic, self.critic_target)                                              # soft update critic target

      return actor_loss.item(), critic_loss.item()





# ================================================================================================================================================
# ================================================================================================================================================




def train_stage1_allocator(n_nodes=5, num_episodes=5000, env_seed=0, max_want=1.0):
   '''
   Train DDPG agent to minimize penalty_short in Stage 1 environment.
   Returns the trained agent.
   
   Parameters:
      - n_nodes: number of nodes in the environment
      - num_episodes: number of training episodes
      - env_seed: random seed for environment
   '''
   env = AllocationEnvStage1(n_nodes=n_nodes, lambda_short=3.0, seed=env_seed, max_want=max_want)            # create Stage 1 env     

   agent = DDPGAgent(                                                                      # create DDPG agent
      state_dim=3 * n_nodes,                                                              # wants + degree_dummy
      n_nodes=n_nodes,
      gamma=0.0,                                                                          # no discounting for single-step episode
      actor_lr=1e-4,
      critic_lr=1e-3,
      tau=0.001,
      batch_size=64,
   )

   returns = []                                                                            # track episode returns
   for episode in range(1, num_episodes + 1):                      
      state = env.reset()
      done = False
      ep_ret = 0.0

      while not done:                                                                     # single-step episode
         noise_scale = max(0.01, 0.3 * (1.0 - episode / num_episodes))                   # decay exploration noise
         action = agent.select_action(state, noise_scale=noise_scale)                    # select action with noise
         next_state, reward, done, info = env.step(action)                               # take env step

         agent.push_transition(state, action, reward, next_state, done)                  # store transition
         actor_loss, critic_loss = agent.train_step()                                    # perform training step

         state = next_state                                                              # update state
         ep_ret += reward                                                                # update episode return

      returns.append(ep_ret)

      if episode % 100 == 0:                                                              # log every 100 episodes
         recent = returns[-100:]
         avg_recent = sum(recent) / len(recent)
         print(
               f"[Stage 1] Episode {episode:5d} | "
               f"Ep reward: {ep_ret: .4f} | "
               f"Mean last-100: {avg_recent: .4f} | "
               f"Actor loss: {actor_loss if actor_loss is not None else 0: .4f} | "
               f"Critic loss: {critic_loss if critic_loss is not None else 0: .4f}"
         )

   print("[Stage 1] Done training allocator on penalty_short.")
   agent.save(
      "checkpoints/stage1_newest_final.pt",
      extra={
         "n_nodes": n_nodes,
         "env_seed": env_seed,
         "max_want": max_want,
      },
   )
   return agent


def evaluate_stage1_greedy_vs_random(
   agent,
   n_nodes,
   wants_fixed,
   env_seed=123,
   lambda_short=3.0,
   n_eval_episodes=100,
   max_want=1.0,
):
   """
   Evaluate trained Stage 1 agent (greedy policy) vs random baseline on fixed wants.
   
   Parameters:
      - agent: trained DDPG agent
      - n_nodes: number of nodes in the environment
      - wants_fixed: fixed wants vector for evaluation
      - env_seed: random seed for environment
      - lambda_short: penalty weight for shortfall
      - n_eval_episodes: number of evaluation episodes
   """
   
   env_greedy = AllocationEnvStage1(n_nodes=n_nodes, lambda_short=lambda_short, seed=env_seed, wants_init=wants_fixed, max_want=max_want,)

   total_ret_greedy = 0.0
   for _ in range(n_eval_episodes):
      state = env_greedy.reset()                                                          # reset env
      done = False
      ep_ret = 0.0
      while not done:                                                                     # run episode with greedy policy
         action = agent.select_action(state, noise_scale=0.0)
         next_state, reward, done, _ = env_greedy.step(action)
         ep_ret += reward
         state = next_state
      total_ret_greedy += ep_ret
   avg_greedy = total_ret_greedy / n_eval_episodes

   env_rand = AllocationEnvStage1(n_nodes=n_nodes, lambda_short=lambda_short, seed=env_seed, wants_init=wants_fixed, max_want=max_want,)

   total_ret_rand = 0.0
   for _ in range(n_eval_episodes):
      state = env_rand.reset()
      done = False
      ep_ret = 0.0
      while not done:
         a = np.random.rand(n_nodes)
         a = project_to_simplex(a)
         next_state, reward, done, _ = env_rand.step(a)
         ep_ret += reward
         state = next_state
      total_ret_rand += ep_ret
   avg_rand = total_ret_rand / n_eval_episodes

   print("Stage 1: Greedy vs Random on fixed wants")
   print(f"  wants_fixed: {np.round(wants_fixed, 3)}")
   print(f"  Avg greedy return over {n_eval_episodes} eps : {avg_greedy:.4f}")
   print(f"  Avg random return over {n_eval_episodes} eps : {avg_rand:.4f}")
   print(f"  Greedy - Random = {avg_greedy - avg_rand:.4f}")




# ================================================================================================================================================
# ================================================================================================================================================




def evaluate_greedy_policy(
   agent,
   env_class,
   n_nodes,
   episode_length,
   env_seed,
   alpha=0.2,
   max_want=1.0,
   n_eval_episodes=50,
   **env_kwargs,
):
   '''
   Evaluate greedy policy of the agent in the specified environment class. 
   Returns average return over evaluation episodes.
   
   Parameters:
      - agent: trained DDPG agent
      - env_class: environment class to evaluate in
      - n_nodes: number of nodes in the environment
      - episode_length: length of each episode
      - env_seed: random seed for environment
      - n_eval_episodes: number of evaluation episodes
      - env_kwargs: additional keyword arguments for environment initialization
   '''
   
   env = env_class(n_nodes=n_nodes, episode_length=episode_length, seed=env_seed, alpha=alpha, max_want=max_want, **env_kwargs,)
   total_return = 0.0
   for _ in range(n_eval_episodes):
      state = env.reset()
      done = False
      ep_ret = 0.0
      while not done:
         action = agent.select_action(state, noise_scale=0.0)
         next_state, reward, done, _ = env.step(action)
         ep_ret += reward
         state = next_state
      total_return += ep_ret
   return total_return / n_eval_episodes




# ================================================================================================================================================
# ================================================================================================================================================




def train_stage2_trust(
   pretrained_actor_state_dict,
   n_nodes=5,
   wants=None,
   trust_testing: Optional[np.ndarray] = None,
   num_episodes=5000,
   episode_length=10,
   env_seed=0,
   graph_type="core_periphery",
   alpha=0.2,
   max_want=1.0,
   graph_jsonl_path: Optional[str] = None,
):
   '''
   Train DDPG agent in Stage 2 environment with trust dynamics.
   
   Parameters:
      - pretrained_actor_state_dict: state dict of pretrained actor from Stage 1
      - n_nodes: number of nodes in the environment
      - wants: fixed wants vector for training and testing
      - trust_testing: fixed trust vector for testing
      - num_episodes: number of training episodes
      - episode_length: length of each episode
      - env_seed: random seed for environment
      - graph_type: type of graph structure in the environment
   '''
   
   if trust_testing is None:
      trust_testing = np.ones(n_nodes) * (1/n_nodes)                                      # default trust vector for testing
   
   graph_sampler = None
   if graph_jsonl_path is not None:
      graph_sampler = GraphSampler(graph_jsonl_path, seed=env_seed, shuffle=True)
      print(f"Using GraphSampler from {graph_jsonl_path}.")
   
   
   env = ResourceAllocTrustEnv(n_nodes=n_nodes, episode_length=episode_length, seed=env_seed, wants_init=wants, graph_type=graph_type, alpha=alpha, max_want=max_want, graph_sampler=graph_sampler, resample_graph_each_reset=True)  # create Stage 2 env
   agent = DDPGAgent(
      state_dim=3 * n_nodes,                                                              # wants + degree_vec
      n_nodes=n_nodes,
      gamma=0.9,                                                                          # discount factor for multi-step episode
      actor_lr=1e-4,
      critic_lr=1e-3,
      tau=0.001,
      batch_size=64,
   )
   
   agent.load_actor_weights(pretrained_actor_state_dict)                                   # load pretrained actor weights from Stage 1
   returns = []
   for episode in range(1, num_episodes + 1):
      # print(
      #    "[train] graph_id=", env.graph_meta.get("graph_id"),
      #    "family=", env.graph_meta.get("family"),
      #    "deg_max=", float(env.degree_vec.max()),
      #    "num_edges=", env.graph.number_of_edges(),
      # )
      state = env.reset()
      done = False
      ep_ret = 0.0

      while not done:
         noise_scale = max(0.01, 0.2 * (1.0 - episode / num_episodes))                   # decay exploration noise
         action = agent.select_action(state, noise_scale=noise_scale)
         next_state, reward, done, info = env.step(action)

         agent.push_transition(state, action, reward, next_state, done)
         actor_loss, critic_loss = agent.train_step()

         state = next_state
         ep_ret += reward

      returns.append(ep_ret)

      if episode % 100 == 0:
         recent = returns[-100:]
         avg_recent = sum(recent) / len(recent)
         print(
               f"[Stage 2] Episode {episode:5d} | "
               f"Ep reward: {ep_ret: .4f} | "
               f"Mean last-100: {avg_recent: .4f} | "
               f"Actor loss: {actor_loss if actor_loss is not None else 0: .4f} | "
               f"Critic loss: {critic_loss if critic_loss is not None else 0: .4f}"
         )

   
   agent.save(
      "checkpoints/stage2_newest_final.pt",
      extra={
         "n_nodes": n_nodes,
         "episode_length": episode_length,
         "env_seed": env_seed,
         "alpha": alpha,
         "max_want": max_want,
         "trained_on_jsonl": graph_jsonl_path,
      },
   )
   
   print("[Stage 2] Testing learned policy (no exploration noise):")
   env = ResourceAllocTrustEnv(n_nodes=n_nodes, episode_length=episode_length, seed=env_seed, wants_init=wants, trust_init=trust_testing, graph_type=graph_type, max_want=max_want, alpha=alpha, graph_sampler=graph_sampler, resample_graph_each_reset=False)
   
   for ep in range(1):                                                                     # run one test episode
      state = env.reset()
      total_reward = 0.0
      print(f"\nTest episode {ep+1}")
      print("-" * 40)
      done = False
      step_idx = 0
      while not done:
         step_idx += 1
         wants_raw = state[:n_nodes].copy()
         print(f" Step {step_idx}: sum(wants) = {wants_raw.sum():.3f}")
         action = agent.select_action(state, noise_scale=0.0)
         next_state, reward, done, info = env.step(action)

         for key in ["wants_t", "alloc", "shortfall", "satisfaction", "trust_t", "trust_next"]:
               if key in info and isinstance(info[key], np.ndarray):
                  print(f"  {key:10s}: {np.round(info[key], 3)}")
         print(f"  reward    : {reward:.4f}\n")

         total_reward += reward
         state = next_state

      print(f"Total episode reward: {total_reward:.4f}")
      print(nx.to_dict_of_dicts(env.graph))
      print("=" * 40)

   print("=" * 60)
   print("Evaluating greedy policy vs random baseline on Stage 2 env...\n")

   avg_greedy = evaluate_greedy_policy(
      agent,                                                                              # greedy policy
      env_class=ResourceAllocTrustEnv,
      n_nodes=n_nodes,
      episode_length=episode_length,
      env_seed=env_seed,
      n_eval_episodes=100,
      graph_type=graph_type,
      wants_init=wants,                                                                   # fixed wants
      trust_init=trust_testing,                                                           # fixed trust
      alpha=alpha,
      max_want=max_want,
      graph_sampler=graph_sampler,   
      resample_graph_each_reset=False,
   )

   env_rand = ResourceAllocTrustEnv(                                                       # random policy
      n_nodes=n_nodes,
      episode_length=episode_length,
      seed=env_seed,
      wants_init=wants,                                                                   # fixed wants
      trust_init=trust_testing,                                                           # fixed trust
      graph_type=graph_type,
      alpha=alpha,
      max_want=max_want,
      graph_sampler=graph_sampler,    
      resample_graph_each_reset=False,
   )
   
   total_return = 0.0
   for _ in range(100):
      s = env_rand.reset()
      done = False
      ep_ret = 0.0
      while not done:
         a = np.random.rand(env_rand.n)
         a = project_to_simplex(a)
         ns, r, done, _ = env_rand.step(a)
         ep_ret += r
         s = ns
      total_return += ep_ret
   avg_random = total_return / 100.0

   print(f"Average greedy-policy return over 100 eps : {avg_greedy:.4f}")
   print(f"Average random-policy return over 100 eps : {avg_random:.4f}")
   print(f"Greedy - Random = {avg_greedy - avg_random:.4f}")
   
   return agent




# ================================================================================================================================================
# ================================================================================================================================================





if __name__ == "__main__":
   n_nodes = 10

   stage1_agent = train_stage1_allocator(n_nodes=n_nodes, num_episodes=10000, env_seed=0, max_want=1.0) # train Stage 1 agent (10000 episodes) (allocator)
   pretrained_actor_state_dict = stage1_agent.actor.state_dict()
   
   # wants = np.array([0.1, 0.2, 0.15, 0.25, 0.3, 0.4, 0.35, 0.45, 0.5, 0.45])               # fixed wants for evaluation
   # evaluate_stage1_greedy_vs_random(agent=stage1_agent, n_nodes=n_nodes, wants_fixed=wants, env_seed=999, lambda_short=3.0, n_eval_episodes=100, max_want=0.5,)

   print("\n" + "=" * 80)
   print(f"Stage 2")
   print("=" * 80)
   
   rng = np.random.default_rng()
   trust_testing = rng.beta(a=2, b=5, size=n_nodes) 

   stage2_agent = train_stage2_trust(                                                  # train Stage 2 agent (5000 episodes) (trust dynamics)
      pretrained_actor_state_dict=pretrained_actor_state_dict,
      # wants=wants,
      wants=None, 
      n_nodes=n_nodes,
      num_episodes=50000,
      episode_length=7,
      env_seed=42,
      graph_type=None,
      alpha= 1/n_nodes,
      max_want=1.0,
      trust_testing=trust_testing,
      graph_jsonl_path="data/graphs_train.jsonl",
   )
 
