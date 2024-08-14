import torch
from gym.wrappers import TimeLimit
from stable_baselines3 import DQN
from torch import nn
from toy_env import ToyEnv, generate_multimodal_demos


class Disc(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        in_dim = 6
        hidden_dim = 64
        out_dim = 1
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, X):
        return self.net(X)


from stable_baselines3.common.callbacks import BaseCallback


class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, verbose: int = 0, disc=None):
        super().__init__(verbose)
        self.set_disc(disc)

    def set_disc(self, disc):
        self.disc = disc

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        # update rewards in the DQN buffer
        NN = self.model.replay_buffer.size()

        obs = self.model.replay_buffer.observations[:NN]
        obs = obs.squeeze(1)
        actions = self.model.replay_buffer.actions[:NN]
        actions = actions.squeeze()
        actions = to_onehot(actions)

        # convert to torch
        obs = torch.from_numpy(obs).float()
        actions = torch.from_numpy(actions).float()

        # concat state and action
        disc_in = torch.cat((obs, actions), dim=1)

        with torch.no_grad():
            disc_rewards = self.disc(disc_in)

        self.model.replay_buffer.rewards[:NN] = disc_rewards


import numpy as np


def to_onehot(arr, n_class=4):
    nn = arr.shape[0]
    ret = np.zeros((nn, n_class))
    ret[np.arange(nn), arr] = 1
    return ret


def train():
    demos = generate_multimodal_demos()
    env = ToyEnv(demos)
    env = TimeLimit(env, 100)

    # polic
    policy = DQN(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        buffer_size=500,
        learning_starts=1,
        exploration_initial_eps=0.05,
        exploration_final_eps=0.05,
        exploration_fraction=1.0,
    )

    # disc
    disc = Disc()

    cb = CustomCallback(disc=disc)

    # training loop
    niter = 1000
    bs = 32
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(disc.parameters(), lr=0.0003)
    for ii in range(niter):

        # sample trajectories
        # train policy
        print("Training policy")
        cb.set_disc(disc)
        policy.learn(100, cb)

        # train disc
        print("training disc")

        # sample expert
        expert_inds = torch.randperm(env.demos[0].shape[0])[:bs]
        expert_obs = env.demos[0][expert_inds]
        expert_obs = torch.from_numpy(expert_obs).float()
        expert_actions = env.demos[1][expert_inds]
        expert_actions = to_onehot(expert_actions)
        expert_actions = torch.from_numpy(expert_actions).float()
        expert_in = torch.cat((expert_obs, expert_actions), 1)

        # sample policy
        policy_batch = policy.replay_buffer.sample(bs)
        policy_obs = policy_batch.observations
        policy_actions = policy_batch.actions
        # policy_actions = to_onehot(policy_actions) #to onehot does not work with 2d! BUG was here!
        policy_actions = to_onehot(policy_actions.squeeze())
        policy_actions = torch.from_numpy(policy_actions).float()
        policy_in = torch.cat((policy_obs, policy_actions), dim=1)

        # train disc
        disc_in = torch.cat((expert_in, policy_in), dim=0)
        disc_labels = torch.cat((torch.ones(expert_in.shape[0]), torch.zeros(policy_in.shape[0])), dim=0)

        # loss
        opt.zero_grad()
        preds = disc(disc_in)
        loss = loss_fn(preds.squeeze(), disc_labels)
        loss.backward()
        opt.step()

        print(ii, loss.item(), sum(pp.norm().item() for pp in disc.parameters()))


if __name__ == "__main__":
    train()
