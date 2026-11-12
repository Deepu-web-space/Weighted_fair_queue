"""SAC agent without EWC (Day 3, Block 2).

One update does three things:
1. Critics learn to predict the target value y.
2. The actor learns to choose weights that score well under the critics,
   with an entropy bonus that keeps its choices varied.
3. Target critics move slowly toward the critics (soft update).

Every number comes from config/base.yaml. Nothing is typed in here.
"""

import copy

import numpy as np
import torch
import torch.nn.functional as F

from wfq.networks import Actor, Critic
from wfq.replay import ReplayBuffer


class SACAgent:
    def __init__(self, sim_config, train_config, seed):
        n = sim_config["n_queues"]
        hidden = train_config["hidden_units"]

        self.n_queues = n
        self.gamma = train_config["gamma"]
        self.tau = train_config["tau"]
        self.alpha = train_config["alpha"]
        self.batch_size = train_config["batch_size"]
        self.warmup_steps = train_config["warmup_steps"]
        self.min_weight = train_config["min_weight"]

        # The seed sets torch's random numbers (network start, action
        # sampling) and numpy's (replay batches). Environment seeds are separate.
        torch.set_num_threads(train_config["torch_threads"])
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

        self.actor = Actor(n, hidden, train_config["min_concentration"])
        self.critic1 = Critic(n, hidden)
        self.critic2 = Critic(n, hidden)

        # Target critics start as exact copies and are never trained directly.
        self.target1 = copy.deepcopy(self.critic1)
        self.target2 = copy.deepcopy(self.critic2)
        for p in list(self.target1.parameters()) + list(self.target2.parameters()):
            p.requires_grad_(False)

        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(), lr=train_config["lr_actor"]
        )
        critic_params = list(self.critic1.parameters()) + list(self.critic2.parameters())
        self.critic_opt = torch.optim.Adam(
            critic_params, lr=train_config["lr_critic"]
        )

        self.buffer = ReplayBuffer(train_config["replay_size"], n, n)

    def _floor(self, w):
        """Keep every weight above min_weight, then rescale to sum to 1.

        The environment refuses zero weights, and log-probabilities
        become undefined at zero.
        """
        w = torch.clamp(w, min=self.min_weight)
        return w / w.sum(dim=-1, keepdim=True)

    def act(self, state, deterministic=False):
        """Return a weight vector (numpy array, sums to 1) for one state."""
        s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                w = self.actor.mean_weights(s)
            else:
                w = self._floor(self.actor(s).sample())
        return w.squeeze(0).numpy()

    def store(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)

    def update(self):
        """One gradient step. Returns None during warm-up, else the losses."""
        needed = max(self.warmup_steps, self.batch_size)
        if self.buffer.size < needed:
            return None

        s, a, r, s2, d = self.buffer.sample(self.batch_size, self.rng)
        s = torch.as_tensor(s)
        a = torch.as_tensor(a)
        r = torch.as_tensor(r)
        s2 = torch.as_tensor(s2)
        d = torch.as_tensor(d)

        # 1. Critic target: reward plus the discounted future value,
        #    using the smaller of the two target critics and the entropy bonus.
        with torch.no_grad():
            dist2 = self.actor(s2)
            a2 = self._floor(dist2.rsample())
            logp2 = dist2.log_prob(a2).unsqueeze(-1)
            q_next = torch.min(self.target1(s2, a2), self.target2(s2, a2))
            y = r + self.gamma * (1.0 - d) * (q_next - self.alpha * logp2)

        # 2. Critic loss: both critics move toward the same target.
        q1 = self.critic1(s, a)
        q2 = self.critic2(s, a)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # 3. Actor loss: prefer weights the critics score highly,
        #    and keep some randomness (the alpha * log-prob term).
        #    Gradients reach the critics too, but they are cleared
        #    before the next critic step.
        dist = self.actor(s)
        a_new = self._floor(dist.rsample())
        logp = dist.log_prob(a_new).unsqueeze(-1)
        q_new = torch.min(self.critic1(s, a_new), self.critic2(s, a_new))
        actor_loss = (self.alpha * logp - q_new).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # 4. Move the target critics slowly toward the critics.
        self._soft_update(self.critic1, self.target1)
        self._soft_update(self.critic2, self.target2)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
        }

    def _soft_update(self, source, target):
        """target = tau * source + (1 - tau) * target."""
        with torch.no_grad():
            for sp, tp in zip(source.parameters(), target.parameters()):
                tp.mul_(1.0 - self.tau).add_(self.tau * sp)
