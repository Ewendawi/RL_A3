import os, time
from typing import Any
import numpy as np
import torch
import gymnasium as gym

class ActorConfig:
    def __init__(self, device_name="cpu"):
        self.input_size = 4
        self.layers_dim = [128]
        self.output_size = 2
        self.learning_rate = 0.001
        self.batch_size = 0
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
        self.batch_size = 0
        self.device = torch.device(device_name)

        self.use_base_line = False
        # gae
        self.use_gae = False
        self.gae_lambda = 0.95

        self.max_episode_length = 2000
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
    
    def should_update(self, done, truncated):
        raise NotImplementedError
    
    def take_action(self, state):
        raise NotImplementedError
    
    def update(self):
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

class RenderActor:
    def __init__(self, path, continuous_action):
        self.model = torch.load(path)
        self.continuous_action = continuous_action

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to("cpu")
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
        action = model.take_action(ob)
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


def train_model(env:gym.Env, model:AbstractPGAlgorithm, time_steps, eval_env, eval_episodes, eval_interval, seed=None):
    time_steps_list = []
    eval_return_list = []
    train_return_list = []
    critic_loss_list = []
    actore_loss_list = []

    state = env.reset(seed=seed)[0]

    episode = 0
    episode_critic_loss_list = []
    episode_actor_loss_list = []
    eposode_train_return_list = []
    eposode_train_return = 0
    for i in range(1, time_steps+1):
        action = model.take_action(state)
        if isinstance(env.action_space, gym.spaces.Discrete):
            action = action[0]
        state_, reward, done, truncated, _ = env.step(action)
        eposode_train_return += reward

        model.store_transition(state, action, reward, state_, done)
        if model.should_update(done, truncated):
            critic_loss, actor_loss = model.update()
            episode_critic_loss_list.append(critic_loss)
            episode_actor_loss_list.append(actor_loss)

        if done or truncated:
            state = env.reset(seed=seed)[0]
            eposode_train_return_list.append(eposode_train_return)
            eposode_train_return = 0
            episode += 1
            if not eval_env:
                print(f"episode:{episode},timestep:{i}/{time_steps},train:{eposode_train_return_list[-1]:.1f}, critic_loss:{episode_critic_loss_list[-1]:.4f}, actor_loss:{episode_actor_loss_list[-1]:.4f}")
        else:
            state = state_

        if eval_env and i % eval_interval == 0:
            time_steps_list.append(i)

            eval_return = evalate_model(eval_env, model, eval_episodes, seed=seed)
            eval_return_list.append(eval_return)

            mean_train_return = np.mean(eposode_train_return_list)
            train_return_list.append(mean_train_return)
            eposode_train_return_list = []
            
            mean_critic_loss = np.mean(episode_critic_loss_list)
            critic_loss_list.append(mean_critic_loss)
            critic_loss_list = []
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
    return result