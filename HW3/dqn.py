# -*- coding:utf-8 -*-
import argparse
import os
import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    """parse arguments. You can add other arguments if needed."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=42,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=500000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=1000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--target-network-frequency", type=int, default=100,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=128,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=0.3,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.05,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.1,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=10000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=10,
        help="the frequency of training")
    args = parser.parse_args()
    args.env_id = "LunarLander-v2"
    return args

def make_env(env_id, seed):
    """construct the gym environment"""
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

class QNetwork(nn.Module):
    """
    The Neural Network:
    Linear -> ReLU -> Linear -> ReLU -> Linear.
    The input dimension is the observation space of the environment.
    The output dimension is the action space of the environment.
    And the hidden dimensions are 120 and 84.
    """
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            # nn.Linear(120, 100),
            # nn.ReLU(),
            # nn.Linear(100, 84),
            # nn.ReLU(),
            nn.Linear(84, env.action_space.n),
        )

    def forward(self, x):
        return self.network(x)

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    """
    Linear interpolation between start_e and end_e
    """
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

if __name__ == "__main__":
    
    """parse the arguments"""
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    """we utilize tensorboard yo log the training process"""
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    """
    Set the random seed for reproducibility.
    Use the device (cpu or cuda) for the computation.
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """
    Construct the environment using the random seed.
    """
    envs = make_env(args.env_id, args.seed)

    """
    Construct the Q-network, the optimizer, the target network.
    Only the parameters of the Q-network need to be optimized.
    The parameters of the Q-network are copied to the target network periodically.
    The optimizer is Adam with the learning rate.
    """
    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    """
    Construct the replay buffer.
    The sampled samples are dropped into ReplayBuffer.
    When the number of samples reaches a certain threshold, an update will be made using the data in ReplayBuffer.
    """
    rb = ReplayBuffer(
        args.buffer_size,
        envs.observation_space,
        envs.action_space,
        device,
        handle_timeout_termination=False,
    )

    """
    The training loop.
    Reset the environment.
    """
    obs = envs.reset()
    for global_step in range(args.total_timesteps):
        
        """
        Linearly decay the epsilon value.
        """
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        
        """
        Select the action using epsilon-greedy.
        """
        if random.random() < epsilon:
            actions = envs.action_space.sample()
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=0).cpu().numpy()
        
        """
        Execute the action in the environment and get the next observation, reward, done, and info.
        """
        next_obs, rewards, dones, infos = envs.step(actions)
        # envs.render() # close render during training
        
        if dones:
            print(f"global_step={global_step}, episodic_return={infos['episode']['r']}")
            writer.add_scalar("charts/episodic_return", infos["episode"]["r"], global_step)
            writer.add_scalar("charts/episodic_length", infos["episode"]["l"], global_step)
        
        """
        Add the transition to the replay buffer.
        """
        rb.add(obs, next_obs, actions, rewards, dones, infos)
        
        """
        Update the observation if the episode is not done.
        Otherwise, reset the environment.
        """
        obs = next_obs if not dones else envs.reset()
        
        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            
            """
            Sample a batch of transitions from the replay buffer.
            """
            data = rb.sample(args.batch_size)
            
            """
            Forced not to calculate the gradients.
            Use MSE loss to update the Q-network.
            """
            with torch.no_grad():
                target_max, _ = target_network(data.next_observations).max(dim=1)
                td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
            old_val = q_network(data.observations).gather(1, data.actions).squeeze()
            loss = F.mse_loss(td_target, old_val)

            """
            Log the loss and the Q-values every 100 steps.
            """
            if global_step % 100 == 0:
                writer.add_scalar("losses/td_loss", loss, global_step)
                writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
            
            """
            Clear the past gradients.
            Back propagate the loss.
            Update the parameters of the Q-network.
            """
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            """
            Update the target network every `target_network_frequency` steps.
            """
            if global_step % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())
    """close the env and tensorboard logger"""

    # show the result of DQN
    for _ in range(10):
        obs = envs.reset()
        envs.render()
        while True:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=0).cpu().numpy()
            next_obs, rewards, dones, infos = envs.step(actions)
            envs.render()
            if dones:
                break
            obs = next_obs
        time.sleep(1)

    envs.close()
    writer.close()