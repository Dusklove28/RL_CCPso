from pathlib import Path

import datetime
import os
import random
from collections import deque

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from rl.DDPG.Prioritized_Replay import Memory
from utils.tensor_utils import to_numpy_array


TORCH_DTYPE = torch.float64


def _as_action_vector(value, size):
    array = np.asarray(value, dtype=np.float64)
    if array.shape == ():
        array = np.full(size, float(array), dtype=np.float64)
    return array.reshape(size)


class Actor(nn.Module):
    def __init__(self, state_shape, action_dim, action_bound, action_shift, units=(400, 300)):
        super().__init__()
        state_dim = int(np.prod(state_shape))
        layers = []
        input_dim = state_dim
        for unit in units:
            layers.append(nn.Linear(input_dim, unit))
            layers.append(nn.LeakyReLU())
            input_dim = unit

        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(input_dim, action_dim)
        self.register_buffer('action_bound', torch.as_tensor(action_bound, dtype=TORCH_DTYPE))
        self.register_buffer('action_shift', torch.as_tensor(action_shift, dtype=TORCH_DTYPE))

    def forward(self, state):
        state = state.reshape(state.shape[0], -1)
        x = self.hidden(state)
        x = torch.tanh(self.output(x))
        x = x * self.action_bound
        return x + self.action_shift


class Critic(nn.Module):
    def __init__(self, state_shape, action_dim, units=(48, 24)):
        super().__init__()
        state_dim = int(np.prod(state_shape))
        layers = []
        input_dim = state_dim + action_dim
        for unit in units:
            layers.append(nn.Linear(input_dim, unit))
            layers.append(nn.LeakyReLU())
            input_dim = unit

        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(input_dim, 1)

    def forward(self, state, action):
        state = state.reshape(state.shape[0], -1)
        x = torch.cat([state, action], dim=-1)
        x = self.hidden(x)
        return self.output(x)


def actor(state_shape, action_dim, action_bound, action_shift, units=(400, 300)):
    return Actor(state_shape, action_dim, action_bound, action_shift, units=units)


def critic(state_shape, action_dim, units=(48, 24)):
    return Critic(state_shape, action_dim, units=units)


def update_target_weights(model, target_model, tau=0.005):
    with torch.no_grad():
        model_parameters = dict(model.named_parameters())
        for name, target_parameter in target_model.named_parameters():
            target_parameter.mul_(1.0 - tau)
            target_parameter.add_(model_parameters[name], alpha=tau)

        model_buffers = dict(model.named_buffers())
        for name, target_buffer in target_model.named_buffers():
            target_buffer.copy_(model_buffers[name])


def _get_env_optimizer_name(env):
    optimizer = getattr(env, 'optimizer', None)
    if optimizer is None:
        optimizer = getattr(env, 'pso_swarm', None)
    if optimizer is None:
        return env.__class__.__name__
    return optimizer.__class__.__name__


def _resolve_device(device):
    requested = os.getenv('RL_DDPG_DEVICE', device or 'auto')
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else 'cpu'

    if requested.startswith('cuda') and not torch.cuda.is_available():
        requested = 'cpu'

    return torch.device(requested)


class OrnsteinUhlenbeckNoise:
    def __init__(self, mu, sigma=0.2, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(
            self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


class NormalNoise:
    def __init__(self, mu, sigma=0.15):
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        return np.random.normal(scale=self.sigma, size=self.mu.shape)

    def reset(self):
        pass


class DDPG:
    def __init__(
            self,
            env,
            discrete=False,
            use_priority=False,
            lr_actor=1e-5,
            lr_critic=1e-3,
            actor_units=(24, 16),
            critic_units=(24, 16),
            noise='norm',
            sigma=0.15,
            tau=0.125,
            gamma=0.85,
            batch_size=64,
            memory_cap=100000,
            device='auto',
    ):
        self.env = env
        self.device = _resolve_device(device)
        self.state_shape = env.observation_space.shape
        self.state_dim = int(np.prod(self.state_shape))
        self.action_dim = env.action_space.n if discrete else env.action_space.shape[0]
        self.discrete = discrete
        self.action_bound = _as_action_vector(
            (env.action_space.high - env.action_space.low) / 2 if not discrete else 1.0,
            self.action_dim,
        )
        self.action_shift = _as_action_vector(
            (env.action_space.high + env.action_space.low) / 2 if not discrete else 0.0,
            self.action_dim,
        )
        self.action_low = self.action_shift - self.action_bound
        self.action_high = self.action_shift + self.action_bound
        self.action_low_tensor = torch.as_tensor(self.action_low, dtype=TORCH_DTYPE, device=self.device).unsqueeze(0)
        self.action_high_tensor = torch.as_tensor(self.action_high, dtype=TORCH_DTYPE, device=self.device).unsqueeze(0)

        self.use_priority = use_priority
        self.memory = Memory(capacity=memory_cap) if use_priority else deque(maxlen=memory_cap)
        if noise == 'ou':
            self.noise = OrnsteinUhlenbeckNoise(mu=np.zeros(self.action_dim), sigma=sigma)
        else:
            self.noise = NormalNoise(mu=np.zeros(self.action_dim), sigma=sigma)

        self.actor = actor(self.state_shape, self.action_dim, self.action_bound, self.action_shift, actor_units).to(
            device=self.device,
            dtype=TORCH_DTYPE,
        )
        self.actor_target = actor(
            self.state_shape,
            self.action_dim,
            self.action_bound,
            self.action_shift,
            actor_units,
        ).to(device=self.device, dtype=TORCH_DTYPE)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)
        update_target_weights(self.actor, self.actor_target, tau=1.0)

        self.critic = critic(self.state_shape, self.action_dim, critic_units).to(
            device=self.device,
            dtype=TORCH_DTYPE,
        )
        self.critic_target = critic(self.state_shape, self.action_dim, critic_units).to(
            device=self.device,
            dtype=TORCH_DTYPE,
        )
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic)
        update_target_weights(self.critic, self.critic_target, tau=1.0)

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.summaries = {}

    def _to_tensor(self, value):
        return torch.as_tensor(np.asarray(value, dtype=np.float64), dtype=TORCH_DTYPE, device=self.device)

    def act(self, state, add_noise=True):
        state_tensor = self._to_tensor(np.expand_dims(state, axis=0))
        self.actor.eval()
        self.critic.eval()
        with torch.no_grad():
            action = self.actor(state_tensor)
            if add_noise:
                noise = torch.as_tensor(
                    self.noise() * self.action_bound,
                    dtype=TORCH_DTYPE,
                    device=self.device,
                ).unsqueeze(0)
                action = action + noise

            action = torch.maximum(torch.minimum(action, self.action_high_tensor), self.action_low_tensor)
            self.summaries['q_val'] = float(self.critic(state_tensor, action)[0][0].detach().cpu().item())

        return action.detach().cpu()

    def save_model(self, a_fn, c_fn):
        actor_path = Path(a_fn)
        critic_path = Path(c_fn)
        actor_path.parent.mkdir(parents=True, exist_ok=True)
        critic_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_actor(self, a_fn):
        actor_state = torch.load(a_fn, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(actor_state)
        self.actor_target.load_state_dict(actor_state)

    def load_critic(self, c_fn):
        critic_state = torch.load(c_fn, map_location=self.device, weights_only=True)
        self.critic.load_state_dict(critic_state)
        self.critic_target.load_state_dict(critic_state)

    def remember(self, state, action, reward, next_state, done):
        action_array = np.asarray(to_numpy_array(action), dtype=np.float64).reshape(-1)
        state_array = np.asarray(state, dtype=np.float64).reshape(-1)
        next_state_array = np.asarray(next_state, dtype=np.float64).reshape(-1)

        if self.use_priority:
            transition = np.hstack([state_array, action_array, reward, next_state_array, done])
            self.memory.store(transition)
        else:
            self.memory.append([
                state_array.reshape(1, -1),
                action_array.reshape(1, -1),
                reward,
                next_state_array.reshape(1, -1),
                done,
            ])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        if self.use_priority:
            tree_idx, samples, is_weights = self.memory.sample(self.batch_size)
            split_shape = np.cumsum([self.state_dim, self.action_dim, 1, self.state_dim])
            states, actions, rewards, next_states, dones = np.hsplit(samples, split_shape)
        else:
            tree_idx = None
            is_weights = np.ones((self.batch_size, 1), dtype=np.float64)
            samples = random.sample(self.memory, self.batch_size)
            states = np.vstack([sample[0] for sample in samples]).astype(np.float64)
            actions = np.vstack([sample[1] for sample in samples]).astype(np.float64)
            rewards = np.asarray([sample[2] for sample in samples], dtype=np.float64).reshape(-1, 1)
            next_states = np.vstack([sample[3] for sample in samples]).astype(np.float64)
            dones = np.asarray([sample[4] for sample in samples], dtype=np.float64).reshape(-1, 1)

        states = self._to_tensor(states)
        actions = self._to_tensor(actions)
        rewards = self._to_tensor(rewards)
        next_states = self._to_tensor(next_states)
        dones = self._to_tensor(dones)
        is_weights = self._to_tensor(is_weights)

        self.actor.train()
        self.critic.train()

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            q_future = self.critic_target(next_states, next_actions)
            target_qs = rewards + q_future * self.gamma * (1.0 - dones)

        q_values = self.critic(states, actions)
        td_error = q_values - target_qs
        critic_loss = torch.mean(is_weights * td_error.pow(2))
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.use_priority:
            abs_errors = torch.sum(torch.abs(td_error), dim=1).detach().cpu().numpy()
            self.memory.batch_update(tree_idx, abs_errors)

        for parameter in self.critic.parameters():
            parameter.requires_grad = False

        actor_actions = self.actor(states)
        actor_loss = -torch.mean(self.critic(states, actor_actions))
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        for parameter in self.critic.parameters():
            parameter.requires_grad = True

        self.summaries['critic_loss'] = float(critic_loss.detach().cpu().item())
        self.summaries['actor_loss'] = float(actor_loss.detach().cpu().item())

        update_target_weights(self.actor, self.actor_target, tau=self.tau)
        update_target_weights(self.critic, self.critic_target, tau=self.tau)

    def train(self, max_episodes=50, max_epochs=8000, max_steps=500, save_freq=50, task_path=None, train_num=0):
        save_freq = 1 if save_freq < 1 else save_freq
        current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        Path('logs').mkdir(exist_ok=True)
        Path('logs').joinpath(f'DDPG_basic_{current_time}').mkdir(exist_ok=True)

        done, episode, steps, epoch, total_reward = False, 0, 0, 0, 0
        cur_state = self.env.reset()
        while episode < max_episodes and epoch < max_epochs:
            if done:
                episode += 1
                print(
                    f"episode {episode}: {total_reward} total reward, {steps} steps, {epoch} epochs "
                    f"optimizer:{_get_env_optimizer_name(self.env)}"
                )

                self.noise.reset()

                if steps >= max_steps and task_path is not None:
                    print(f'episode {episode}, reached max steps')
                    self.save_model(
                        task_path.joinpath(f'ddpg_actor_episode{episode}_round{train_num}.pth'),
                        task_path.joinpath(f'ddpg_critic_episode{episode}_round{train_num}.pth'),
                    )

                done, cur_state, steps, total_reward = False, self.env.reset(), 0, 0
                if episode % save_freq == 0 and task_path is not None:
                    self.save_model(
                        task_path.joinpath(f'ddpg_actor_episode{episode}_round{train_num}.pth'),
                        task_path.joinpath(f'ddpg_critic_episode{episode}_round{train_num}.pth'),
                    )

            action_tensor = self.act(cur_state)
            action = np.argmax(to_numpy_array(action_tensor)) if self.discrete else to_numpy_array(action_tensor[0])
            next_state, reward, done, _ = self.env.step(action)

            self.remember(cur_state, action_tensor, reward, next_state, done)
            self.replay()

            cur_state = next_state
            total_reward += reward
            steps += 1
            epoch += 1

        if task_path is not None:
            self.save_model(
                task_path.joinpath(f'ddpg_actor_final_round{train_num}.pth'),
                task_path.joinpath(f'ddpg_critic_final_round{train_num}.pth'),
            )

    def policy(self, state):
        action = self.act(state, add_noise=False)
        return action[0]

    def test(self, render=True, fps=30, filename='test_render.mp4'):
        cur_state, done, rewards = self.env.reset(), False, 0
        step_num = 0
        while not done:
            step_num += 1
            action_tensor = self.act(cur_state, add_noise=False)
            action = np.argmax(to_numpy_array(action_tensor)) if self.discrete else to_numpy_array(action_tensor[0])
            next_state, reward, done, _ = self.env.step(action)
            cur_state = next_state
            rewards += reward
        return rewards, step_num

    def plot_graph(self):
        return None


if __name__ == '__main__':
    import gym

    gym_env = gym.make('CartPole-v1')
    try:
        assert (gym_env.action_space.high == -gym_env.action_space.low)
        is_discrete = False
        print('Continuous Action Space')
    except AttributeError:
        is_discrete = True
        print('Discrete Action Space')

    ddpg = DDPG(gym_env, discrete=is_discrete)
    ddpg.train(max_episodes=1000)
