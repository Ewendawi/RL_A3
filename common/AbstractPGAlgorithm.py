import os, time
from typing import Any
import numpy as np
import torch
import gymnasium as gym
from tensorboardX import SummaryWriter

class ActorConfig:
    def __init__(self, device_name="cpu"):
        self.input_size = 4
        self.layers_dim = [128]
        self.output_size = 2
        self.learning_rate = 0.001
        self.batch_size = 64
        self.device = torch.device(device_name)

        self.entropy_weight = 0.0
        self.gsde_noise_scale = 0.0

        self.continuous_action_space = False
        self.action_low_bound = 1
        self.action_uppper_bound = 1
        self.action_bound = 1

class AbstractActor:
    def __init__(self, config: ActorConfig):
        self.config = config

    def take_action(self, state):
        raise NotImplementedError

    def update(self, samples, psi_values):
        raise NotImplementedError

    def save(self, name):
        raise NotImplementedError
    
class CriticConfig:
    def __init__(self, device_name="cpu"):
        self.gamma = 0.98

        # value network
        self.input_size = 4
        self.layers_dim = [128]
        self.learning_rate = 0.01
        self.batch_size = 64
        self.device = torch.device(device_name)

        self.use_base_line = False
        # gae
        self.use_gae = False
        self.gae_lambda = 0.95

        self.max_episode_length = 2048
        self.n_steps = 0 # 0 means MC 

        # reward standardization
        self.standardize_reward = False 
        self.reward_uppper_bound = 0.0
        self.reward_low_bound = -16.0

class AbstractCritic:
    def __init__(self, config: CriticConfig):
        self.config = config

    def update(self, samples):
        raise NotImplementedError


class AbstractPGAlgorithm:
    def __init__(self, actor_config: ActorConfig, critic_config: CriticConfig):
        pass
    
    def store_transition(self, state, action, reward, next_state, done):
        raise NotImplementedError

    def create_sample_buffer(self):
        raise NotImplementedError
    
    def should_update(self, done, truncated):
        raise NotImplementedError

    def should_update_with_buffer(self, done, truncated, sample_buffer):
        raise NotImplementedError
    
    def take_action(self, state):
        raise NotImplementedError
    
    def update(self):
        raise NotImplementedError

    def update_with_buffer(self, sample_buffer):
        raise NotImplementedError

    def save(self, name):
        raise NotImplementedError
    
def save_model(model, name):
    plots_dir = "./models"
    if not os.path.isdir(plots_dir):
        os.makedirs(plots_dir)
    now = time.strftime("%m%d_%H_%M_%S", time.localtime(time.time()))
    path = f"{plots_dir}/{name}-{now}"
    torch.save(model, path)
    return path

class RenderActor:
    def __init__(self, path, continuous_action):
        self.model = torch.load(path)
        self.continuous_action = continuous_action

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).view(1,-1).to("cpu")
        with torch.no_grad():
            if self.continuous_action:
                mu, std = self.model(state)
                dist = torch.distributions.Normal(mu, std)
                action = dist.sample()
            else:
                probs = self.model(state)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
        return action.tolist()

def render_env(path):
    file_name = os.path.basename(path)
    name_parts = file_name.split("_")
    env_name = name_parts[1]
    
    env = gym.make(env_name, render_mode="human")
    continuous_action = isinstance(env.action_space, gym.spaces.Box)
    model = RenderActor(path, continuous_action)
    
    ob = env.reset()[0]
    r = 0
    while True:
        action = model.take_action(ob)[0]
        ob, reward, done, truncated, info = env.step(action)
        env.render()
        r += reward
        if done or truncated:
            ob = env.reset()[0]
            print(r)
            r = 0

def evalate_model(env:gym.Env, model:AbstractPGAlgorithm, episodes=20, seed=None) -> np.floating[Any]:
    return_list = []

    for _ in range(episodes):
        episode_return = 0
        state = env.reset(seed=seed)[0]

        while True:
            action = model.take_action(state)
            if isinstance(env.action_space, gym.spaces.Discrete):
                action = action[0]
            state_, reward, done, truncated, _ = env.step(action)

            episode_return += reward
            
            if done or truncated:
                return_list.append(episode_return)
                break
            else:
                state = state_
    return np.mean(return_list)


def train_model(env:gym.Env, model:AbstractPGAlgorithm, time_steps, eval_env, eval_episodes, eval_interval, seed=None, tensorboard_dir=None):
    time_steps_list = []
    eval_return_list = []
    train_return_list = []
    critic_loss_list = []
    actore_loss_list = []
    approx_kl_div = []

    writer = None
    if tensorboard_dir:
        now = time.strftime("%m%d_%H_%M_%S", time.localtime(time.time()))
        dir_path = f"./logs/{tensorboard_dir}_{now}"
        writer = SummaryWriter(dir_path)
    def write_to_tensorboard(key, value, i):
        if writer:
            writer.add_scalar(key, value, i)

    state = env.reset(seed=seed)[0]

    episode = 0
    episode_critic_loss_list = []
    episode_actor_loss_list = []
    episode_train_return_list = []
    episode_approx_kl_div_list = []
    episode_train_return = 0
    episode_length = 0
    for i in range(1, time_steps+1):
        action = model.take_action(state)
        if isinstance(env.action_space, gym.spaces.Discrete):
            action = action[0]
        state_, reward, done, truncated, _ = env.step(action)
        episode_train_return += reward
        episode_length += 1

        model.store_transition(state, action, reward, state_, done)
        if model.should_update(done, truncated):
            critic_loss, actor_loss, log_infos = model.update()
            episode_critic_loss_list.append(critic_loss)
            episode_actor_loss_list.append(actor_loss)
            write_to_tensorboard("train/policy_gradient_loss", actor_loss, i)
            write_to_tensorboard("train/value_loss", critic_loss, i)
            for key, value in log_infos.items():
                write_to_tensorboard(f"train/{key}", value, i)
                if key == "approx_kl_div":
                    episode_approx_kl_div_list.append(value)

        if done or truncated:
            state = env.reset(seed=seed)[0]
            write_to_tensorboard("rollout/ep_rew_mean", episode_train_return, i)
            write_to_tensorboard("rollout/ep_len_mean", episode_length, i)
            episode_train_return_list.append(episode_train_return)
            episode_train_return = 0
            episode_length = 0
            episode += 1
        else:
            state = state_

        if i % eval_interval == 0 or i == time_steps:
            time_steps_list.append(i)

            eval_return = 0
            if eval_env:
                eval_return = evalate_model(eval_env, model, eval_episodes, seed=seed)
                write_to_tensorboard("eval/return", eval_return, i)
            eval_return_list.append(eval_return)

            mean_train_return = np.mean(episode_train_return_list)
            train_return_list.append(mean_train_return)
            episode_train_return_list = []

            if len(episode_approx_kl_div_list) > 0:
                mean_approx_kl_div = np.mean(episode_approx_kl_div_list)
                approx_kl_div.append(mean_approx_kl_div)
                episode_approx_kl_div_list = []
            
            mean_critic_loss = np.mean(episode_critic_loss_list)
            critic_loss_list.append(mean_critic_loss)
            episode_critic_loss_list = []
            
            mean_actor_loss = np.mean(episode_actor_loss_list)
            actore_loss_list.append(mean_actor_loss)
            episode_critic_loss_list = []

            print(f"episode:{episode},timestep:{i}/{time_steps},train:{mean_train_return:.1f}, eval:{eval_return:.1f}, critic_loss:{mean_critic_loss:.4f}, actor_loss:{mean_actor_loss:.4f}")

    result = {
        "eval": eval_return_list,
        "train": train_return_list,
        "critic_loss": critic_loss_list,
        "actor_loss": actore_loss_list,
        "time_steps": time_steps_list,
    }
    if len(approx_kl_div) > 0:
        result["approx_kl_div"] = approx_kl_div
    return result

class trainEnv:
    def __init__(self, env, seed=None): 
        self.env = env
        self.seed = seed

        self.returns = []
        self.episode_lengths = []
        self.actor_losses = []
        self.critic_losses = []

        self.episode_return = 0
        self.episode_length = 0

        self.buffer = None
        self.current_state = None

    def step_update(self, state, action, reward, next_state, done):
        self.episode_return += reward
        self.episode_length += 1
        self.buffer.store_transition(state, action, reward, next_state, done)

    def flush_episode(self, writer, i):
        if writer:
            writer.add_scalar("rollout/ep_rew_mean", self.episode_return, i)
            writer.add_scalar("rollout/ep_len_mean", self.episode_length, i)
        
        self.returns.append(self.episode_return)
        self.episode_lengths.append(self.episode_length)
        self.episode_return = 0
        self.episode_length = 0

    def flush_loss(self, writer, i, actor_loss, critic_loss):
        if writer:
            writer.add_scalar("train/policy_gradient_loss", actor_loss, i)
            writer.add_scalar("train/value_loss", critic_loss, i)
        self.actor_losses.append(actor_loss)
        self.critic_losses.append(critic_loss)

    def flush_all(self):
        mean_return = np.mean(self.returns)
        mean_length = np.mean(self.episode_lengths)
        mean_actor_loss = np.mean(self.actor_losses)
        mean_critic_loss = np.mean(self.critic_losses)

        self.returns = []
        self.episode_lengths = []
        self.actor_losses = []
        self.critic_losses = []

        return mean_return, mean_length, mean_actor_loss, mean_critic_loss

    def reset_env(self):
        return self.env.reset(seed=self.seed)[0]

    @classmethod
    def flush_envs(cls, envs):
        returns = []
        lengths = []
        actor_losses = []
        critic_losses = []
        for env in envs:
            mean_return, mean_length, mean_actor_loss, mean_critic_loss = env.flush_all()
            returns.append(mean_return)
            lengths.append(mean_length)
            actor_losses.append(mean_actor_loss)
            critic_losses.append(mean_critic_loss)
        return np.mean(returns), np.mean(lengths), np.mean(actor_losses), np.mean(critic_losses)

def train_model_with_vectors(envs:[gym.Env], model:AbstractPGAlgorithm, time_steps, eval_env, eval_episodes, eval_interval, seed=None, tensorboard_dir=None):
    time_steps_list = []
    eval_return_list = []
    train_return_list = []
    critic_loss_list = []
    actore_loss_list = []

    writer = None
    if tensorboard_dir:
        now = time.strftime("%m%d_%H_%M_%S", time.localtime(time.time()))
        dir_path = f"./logs/{tensorboard_dir}_{now}"
        writer = SummaryWriter(dir_path)

    num_envs = len(envs)
    train_envs = []
    for env in envs:
        train_env = trainEnv(env, seed+1) 
        train_env.buffer = model.create_sample_buffer()
        train_envs.append(train_env)
        
    episode = 0
    for i in range(0, time_steps//num_envs):
        step = i*num_envs
        for train_env in train_envs:
            if i == 0:
                train_env.current_state = train_env.reset_env()
            action = model.take_action(train_env.current_state)
            if isinstance(env.action_space, gym.spaces.Discrete):
                action = action[0]
            state_, reward, done, truncated, _ = train_env.env.step(action)
            train_env.step_update(train_env.current_state, action, reward, state_, done)
            if model.should_update_with_buffer(done, truncated, train_env.buffer):
                critic_loss, actor_loss, log_infos = model.update_with_buffer(train_env.buffer)
                train_env.flush_loss(writer, step, actor_loss, critic_loss)
                for key, value in log_infos.items():
                    if writer:
                        writer.add_scalar(f"train/{key}", value, step)
                
            if done or truncated:
                train_env.current_state = train_env.reset_env()
                train_env.flush_episode(writer, step)
                episode += 1
            else:
                train_env.current_state = state_

        if step % eval_interval == 0:
            time_steps_list.append(i)

            eval_return = 0
            if eval_env:
                eval_return = evalate_model(eval_env, model, eval_episodes, seed)
                if writer:
                    writer.add_scalar("eval/return", eval_return, step)
            eval_return_list.append(eval_return)

            mean_train_return, mean_length, mean_actor_loss, mean_critic_loss = trainEnv.flush_envs(train_envs)
            train_return_list.append(mean_train_return)
            critic_loss_list.append(mean_critic_loss)
            actore_loss_list.append(mean_actor_loss)

            print(f"episode:{episode},timestep:{step}/{time_steps},train:{mean_train_return:.1f}, eval:{eval_return:.1f}, critic_loss:{mean_critic_loss:.4f}, actor_loss:{mean_actor_loss:.4f}")

    result = {
        "eval": eval_return_list,
        "train": train_return_list,
        "critic_loss": critic_loss_list,
        "actor_loss": actore_loss_list,
        "time_steps": time_steps_list,
    }
    return result