"""
Adapted from Pytorch's "Train a Mario-playing RL Agent" (**Authors:** `Yuansong Feng <https://github.com/YuansongFeng>`__, `Suraj Subramanian <https://github.com/suraj813>`__, `Howard Wang <https://github.com/hw26>`__, `Steven Guo <https://github.com/GuoYuzhang>`__.)
"""

import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, os

# Gym is an OpenAI toolkit for RL
import gymnasium as gym
from gym.spaces import Box
from gym.wrappers import FrameStack

# Our bot environment, built with the openai_ros package
import bot_env

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

######################################################################
# RL Definitions
# """"""""""""""""""
#
# **Environment** The world that an agent interacts with and learns from.
#
# **Action** :math:`a` : How the Agent responds to the Environment. The
# set of all possible Actions is called *action-space*.
#
# **State** :math:`s` : The current characteristic of the Environment. The
# set of all possible States the Environment can be in is called
# *state-space*.
#
# **Reward** :math:`r` : Reward is the key feedback from Environment to
# Agent. It is what drives the Agent to learn and to change its future
# action. An aggregation of rewards over multiple time steps is called
# **Return**.
#
# **Optimal Action-Value function** :math:`Q^*(s,a)` : Gives the expected
# return if you start in state :math:`s`, take an arbitrary action
# :math:`a`, and then for each future time step take the action that
# maximizes returns. :math:`Q` can be said to stand for the “quality” of
# the action in a state. We try to approximate this function.
#


######################################################################
# Environment
# """"""""""""""""
#
# Initialize Environment
# ------------------------
#

# Initialize bot environment
env = BotEnv()

env.reset()
next_state, reward, done, trunc, info = env.step(action=0)
print(f"{next_state.shape},\n {reward},\n {done},\n {info}")


######################################################################
# Preprocess Environment
# ------------------------
#
# The view from the robot's camera is returned in next_state. This
# contains more data than we can afford to process, so it needs to be
# downsampled first.
#
# We use **Wrappers** to preprocess environment data before sending it to
# the agent.
#
# ``ResizeObservation`` downsamples each observation into a square image.
# The new size of each observation is: ``[3, 84, 84]`` (84 x 84 is the new
# width and height, and 3 is for the RGB values).
#
# ``SkipFrame`` is a custom wrapper that inherits from ``gym.Wrapper`` and
# implements the ``step()`` function. Because consecutive frames don’t
# vary much, we can skip n-intermediate frames without losing much
# information. The n-th frame aggregates rewards accumulated over each
# skipped frame.
#


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunk, info


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape, antialias=True), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation


# Apply Wrappers to environment
env = SkipFrame(env, skip=4)
env = ResizeObservation(env, shape=84)


######################################################################
# Agent
# """""""""
#
# We create a class ``Bot`` to represent our agent in the game. The bot
# should be able to:
#
# -  **Act** according to the optimal action policy based on the current
#    state (of the environment).
#
# -  **Remember** experiences. Experience = (current state, current
#    action, reward, next state). The bot *caches* and later *recalls*
#    its experiences to update its action policy.
#
# -  **Learn** a better action policy over time
#


class Bot:
    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # The bot's DNN to predict the most optimal action
        self.net = BotNet(self.state_dim, self.action_dim).float()
        self.net = self.net.to(device=self.device)

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        self.save_every = 5e5  # no. of experiences between saving BotNet
        
        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(100000, device=torch.device("cpu")))
        self.batch_size = 32
        
        self.gamma = 0.9
        
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()
        
        self.burnin = 1e4  # min. experiences before training
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync


    ######################################################################
    # Act
    # --------------
    #
    # For any given state, an agent can choose to do the most optimal action
    # (**exploit**) or a random action (**explore**).
    #
    # The bot randomly explores with a chance of ``self.exploration_rate``; when
    # it chooses to exploit, it relies on ``BotNet`` to provide the most optimal
    # action.
    #

    def act(self, state):
        """
        Given a state, choose an epsilon-greedy action and update value of step.

        Inputs:
        state(``LazyFrame``): A single observation of the current state, dimension is (state_dim)
        Outputs:
        ``action_idx`` (``int``): An integer representing which action the bot will perform
        """
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx


    ######################################################################
    # Cache and Recall
    # ----------------------
    #
    # These two functions serve as the bot’s “memory” process.
    #
    # ``cache()``: Each time the bot performs an action, it stores the
    # ``experience`` to its memory. Its experience includes the current
    # *state*, *action* performed, *reward* from the action, the *next state*,
    # and whether the task is *done*.
    #
    # ``recall()``: The bot randomly samples a batch of experiences from its
    # memory, and uses that to learn how to complete the task.
    #

    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (``LazyFrame``),
        next_state (``LazyFrame``),
        action (``int``),
        reward (``float``),
        done(``bool``))
        """
        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x
        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()

        state = torch.tensor(state)
        next_state = torch.tensor(next_state)
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        done = torch.tensor([done])

        # self.memory.append((state, next_state, action, reward, done,))
        self.memory.add(TensorDict({"state": state, "next_state": next_state, "action": action, "reward": reward, "done": done}, batch_size=[]))

    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = self.memory.sample(self.batch_size).to(self.device)
        state, next_state, action, reward, done = (batch.get(key) for key in ("state", "next_state", "action", "reward", "done"))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()


    ######################################################################
    # TD Estimate & TD Target
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~
    #
    # Two values are involved in learning:
    #
    # **TD Estimate** - the predicted optimal :math:`Q^*` for a given state
    # :math:`s`
    #
    # .. math::
    #
    #
    #    {TD}_e = Q_{online}^*(s,a)
    #
    # **TD Target** - aggregation of current reward and the estimated
    # :math:`Q^*` in the next state :math:`s'`
    #
    # .. math::
    #
    #
    #    a' = argmax_{a} Q_{online}(s', a)
    #
    # .. math::
    #
    #
    #    {TD}_t = r + \gamma Q_{target}^*(s',a')
    #
    # Because we don’t know what next action :math:`a'` will be, we use the
    # action :math:`a'` maximizes :math:`Q_{online}` in the next state
    # :math:`s'`.
    #
    # Notice we use the
    # `@torch.no_grad() <https://pytorch.org/docs/stable/generated/torch.no_grad.html#no-grad>`__
    # decorator on ``td_target()`` to disable gradient calculations here
    # (because we don’t need to backpropagate on :math:`\theta_{target}`).
    #

    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()


    ######################################################################
    # Updating the model
    # ~~~~~~~~~~~~~~~~~~~~~~
    #
    # As the bot samples inputs from its replay buffer, we compute :math:`TD_t`
    # and :math:`TD_e` and backpropagate this loss down :math:`Q_{online}` to
    # update its parameters :math:`\theta_{online}` (:math:`\alpha` is the
    # learning rate ``lr`` passed to the ``optimizer``)
    #
    # .. math::
    #
    #
    #    \theta_{online} \leftarrow \theta_{online} + \alpha \nabla(TD_e - TD_t)
    #
    # :math:`\theta_{target}` does not update through backpropagation.
    # Instead, we periodically copy :math:`\theta_{online}` to
    # :math:`\theta_{target}`
    #
    # .. math::
    #
    #
    #    \theta_{target} \leftarrow \theta_{online}
    #
    #

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())


    ######################################################################
    # Save checkpoint
    # ~~~~~~~~~~~~~~~~~~
    #

    def save(self):
        save_path = (
            self.save_dir / f"bot_net_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"BotNet saved to {save_path} at step {self.curr_step}")


    ######################################################################
    # Putting it all together
    # ~~~~~~~~~~~~~~~~~~
    #

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)


######################################################################
# Learn
# --------------
#
# The bot uses the `DDQN algorithm <https://arxiv.org/pdf/1509.06461>`__
# under the hood. DDQN uses two ConvNets - :math:`Q_{online}` and
# :math:`Q_{target}` - that independently approximate the optimal
# action-value function.
#
# In our implementation, we share feature generator ``features`` across
# :math:`Q_{online}` and :math:`Q_{target}`, but maintain separate FC
# classifiers for each. :math:`\theta_{target}` (the parameters of
# :math:`Q_{target}`) is frozen to prevent updating by backprop. Instead,
# it is periodically synced with :math:`\theta_{online}` (more on this
# later).
#
# Neural Network
# ~~~~~~~~~~~~~~~~~~


class BotNet(nn.Module):
    """mini CNN structure
  input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
  """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.online = self.__build_cnn(c, output_dim)

        self.target = self.__build_cnn(c, output_dim)
        self.target.load_state_dict(self.online.state_dict())

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)

    def __build_cnn(self, c, output_dim):
        return nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )


######################################################################
# Logging
# --------------
#

import numpy as np
import time, datetime
import matplotlib.pyplot as plt


class MetricLogger:
    def __init__(self, save_dir):
        self.save_log = save_dir / "log"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )
        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"

        # History metrics
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        # Moving averages, added for every call to record()
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        # Current episode metric
        self.init_episode()

        # Timing
        self.record_time = time.time()

    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self):
        "Mark end of episode"
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)

        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, epsilon, step):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {episode} - "
            f"Step {step} - "
            f"Epsilon {epsilon} - "
            f"Mean Reward {mean_ep_reward} - "
            f"Mean Length {mean_ep_length} - "
            f"Mean Loss {mean_ep_loss} - "
            f"Mean Q Value {mean_ep_q} - "
            f"Time Delta {time_since_last_record} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in ["ep_lengths", "ep_avg_losses", "ep_avg_qs", "ep_rewards"]:
            plt.clf()
            plt.plot(getattr(self, f"moving_avg_{metric}"), label=f"moving_avg_{metric}")
            plt.legend()
            plt.savefig(getattr(self, f"{metric}_plot"))


######################################################################
# Let’s play!
# """""""""""""""
#
# The code is currently set to 40 episodes to test, but this will probably
# need to be much higher for the resulting neural network to be effective -
# the mario example suggested 40,000 episodes, but that task isn't completely
# comparable.
#

use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")
print()

save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

bot = Bot(state_dim=(3, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)

logger = MetricLogger(save_dir)

episodes = 40
for e in range(episodes):

    state = env.reset()

    # Play the game!
    while True:

        # Run agent on the state
        action = bot.act(state)

        # Agent performs action
        next_state, reward, done, trunc, info = env.step(action)

        # Remember
        bot.cache(state, next_state, action, reward, done)

        # Learn
        q, loss = bot.learn()

        # Logging
        logger.log_step(reward, loss, q)

        # Update state
        state = next_state

        # Check if end of game
        if done:
            break

    logger.log_episode()

    if (e % 20 == 0) or (e == episodes - 1):
        logger.record(episode=e, epsilon=bot.exploration_rate, step=bot.curr_step)
