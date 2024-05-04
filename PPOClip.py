

import numpy as np
import torch
import gymnasium as gym
from common.AbstractPGAlgorithm import AbstractPGAlgorithm, CriticConfig, ActorConfig 
from AC import StochasticActor, NormalCritic, GAE_estimate
from common.helper import SampleBuffer
from common.expConfig import ExpConfig

class PPOClipConfig(CriticConfig):
    def __init__(self, device_name="cpu"):
        super(PPOClipConfig, self).__init__(device_name)
        self.clip = 0.2
        self.dual_clip = 5.0
        self.target_kl = None
        self.recompute_gae = False
        self.use_total_loss = True
        self.value_loss_weight = 0.5
        self.repeat = 10

class PPOCritic(NormalCritic):
    def __init__(self, config: PPOClipConfig):
        super().__init__(config)
        self.config = config

    def updateWithActor(self, samples, actor: StochasticActor):
        g_states, g_actions, g_rewards, g_states_, g_dones = samples
        clip = torch.tensor(self.config.clip).to(self.config.device)

        # returns = self.returns_of_episode(rewards, states, states_, dones).detach()
        g_values = self.value_net(g_states).detach()
        if not self.config.recompute_gae:
            g_gae = GAE_estimate(g_rewards, g_values, g_dones, self.config.gamma, self.config.gae_lambda)
            g_gae = g_gae.view(-1,1).detach()
        g_returns = g_gae + g_values

        dists = actor.action_distribution(g_states)
        g_base_log_probs = actor.log_probs_with_dists(dists, g_actions).detach()
        
        actor_loss_list = []
        value_loss_list = []
        log_infos = {
            "clip_fraction": [],
            "approx_kl": []
        }
        batch_indeices = np.arange(g_states.shape[0])
        stop_training = False
        for _ in range(self.config.repeat):
            np.random.shuffle(batch_indeices)
            batch_size = self.config.batch_size
            batch_size = len(batch_indeices) if batch_size == 0 else batch_size
            for i in range(0, len(batch_indeices), batch_size):
                states = g_states[batch_indeices[i:i+batch_size]]
                actions = g_actions[batch_indeices[i:i+batch_size]]
                rewards = g_rewards[batch_indeices[i:i+batch_size]]
                dones = g_dones[batch_indeices[i:i+batch_size]]
                returns = g_returns[batch_indeices[i:i+batch_size]]
                base_log_probs = g_base_log_probs[batch_indeices[i:i+batch_size]]
                gae = g_gae[batch_indeices[i:i+batch_size]]
                
                dists = actor.action_distribution(states)
                log_probs = actor.log_probs_with_dists(dists, actions) 

                if self.config.recompute_gae:
                    gae = GAE_estimate(rewards, self.value_net(states), dones, self.config.gamma, self.config.gae_lambda)
                    gae = gae.view(-1,1).detach()

                gae = (gae - gae.mean()) / (gae.std() + 1e-8)

                ratio = torch.exp(log_probs - base_log_probs)
                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip).float()).item()
                log_infos["clip_fraction"].append(clip_fraction)
                clipped_ratio_loss = torch.clamp(ratio, 1 - clip, 1 + clip) * gae
                ratio_loss = ratio * gae
                if self.config.dual_clip >= 1:
                    clip1 = torch.min(clipped_ratio_loss, ratio_loss)
                    clip2 = torch.max(clip1, self.config.dual_clip * gae) 
                    actor_loss = -torch.where(gae < 0, clip2, clip1).mean()
                else:
                    actor_loss = -torch.min(clipped_ratio_loss, ratio_loss).mean()

                with torch.no_grad():
                    log_ratio = log_probs - base_log_probs
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).detach().cpu().item()
                    log_infos["approx_kl"].append(approx_kl_div)

                if self.config.target_kl is not None and approx_kl_div > self.config.target_kl:
                    stop_training = True
                    break

                actor_loss = actor_loss - actor.config.entropy_weight * dists.entropy().mean()
                state_values = self.value_net(states)
                value_loss = torch.nn.functional.mse_loss(state_values, returns) 

                actor.optimizer.zero_grad()
                self.value_net_optimizer.zero_grad()
                if self.config.use_total_loss:
                    loss = actor_loss + value_loss * self.config.value_loss_weight
                    loss.backward()
                else:
                    actor_loss.backward()
                    value_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.policy_net.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
                actor.optimizer.step()
                self.value_net_optimizer.step()

                actor_loss_list.append(actor_loss.detach().cpu())
                value_loss_list.append(value_loss.detach().cpu())
            if stop_training:
                break

        for k in log_infos.keys():
            log_infos[k] = np.mean(log_infos[k])
        return np.mean(value_loss_list), np.mean(actor_loss_list), log_infos

# tricks
# - dual clip
# - recompute gae in each repeat

class PPOClip(AbstractPGAlgorithm):
    def __init__(self, actor_config, critic_config: PPOClipConfig):
        super(PPOClip, self).__init__(actor_config, critic_config)
        self.actor = StochasticActor(actor_config)
        self.critic = PPOCritic(critic_config)
        self.sample_buffer = SampleBuffer()
    
    def save(self, name):
        self.actor.save(name)

    def take_action(self, state):
        return self.actor.take_action(state)

    def store_transition(self, state, action, reward, next_state, done):
        self.sample_buffer.store_transition(state, action, reward, next_state, done)

    def create_sample_buffer(self):
        return SampleBuffer()

    def should_update(self, done, truncated):
        return self.should_update_with_buffer(done, truncated, self.sample_buffer)

    def should_update_with_buffer(self, done, truncated, sample_buffer):
        # if done or truncated:
        #     return True
        if sample_buffer.size() >= self.critic.config.max_episode_length:
            return True
        return False

    def update(self):
        return self.update_with_buffer(self.sample_buffer)

    def update_with_buffer(self, sample_buffer):
        samples = sample_buffer.get_all_samples()
        states, actions, rewards, states_, dones = samples
        states = torch.tensor(states, dtype=torch.float32).to(self.actor.config.device)
        actions = torch.tensor(actions).to(self.actor.config.device)
        if len(actions.shape) == 1:
            actions = actions.view(-1,1)
        rewards = torch.tensor(rewards, dtype=torch.float32).view(-1,1).to(self.actor.config.device)
        states_ = torch.tensor(states_, dtype=torch.float32).to(self.actor.config.device)
        dones = torch.tensor(dones, dtype=torch.float32).view(-1,1).to(self.actor.config.device)

        if self.critic.config.standardize_reward:
            mean_reward = (self.critic.config.reward_uppper_bound - self.critic.config.reward_low_bound)/2
            rewards = (rewards + mean_reward) / mean_reward

        samples = (states, actions, rewards, states_, dones)
        critic_loss, actor_loss, log_infos = self.critic.updateWithActor(samples, self.actor)

        sample_buffer.reset()
        return critic_loss, actor_loss, log_infos


def exp_config_for_PPOClip(exp_name="ppo", env_name="", repeat=1, timesteps=20000, device_name="cpu"):
    seed = 12356

    env = gym.make(env_name)
    continuous_action = isinstance(env.action_space, gym.spaces.Box)
    
    actor_config = ActorConfig(device_name=device_name)
    actor_config.input_size = env.observation_space.shape[0]
    if continuous_action:
        actor_config.output_size = env.action_space.shape[0]
        actor_config.action_low_bound = env.action_space.low
        actor_config.action_uppper_bound = env.action_space.high
        actor_config.action_bound = (env.action_space.high - env.action_space.low) / 2
    else:
        actor_config.output_size = env.action_space.n
    actor_config.continuous_action_space = continuous_action
    actor_config.learning_rate = 1e-3

    critic_config = PPOClipConfig(device_name=device_name)
    critic_config.input_size = env.observation_space.shape[0]
    critic_config.n_steps = 1
    critic_config.learning_rate = 1e-2
    critic_config.gamma = 0.9
    critic_config.gae_lambda = 0.9

    eval_interval = 2000
    eval_env = gym.make(env_name)
    eval_episodes = 5

    exp_config = ExpConfig(exp_name, repeat, env, timesteps, eval_interval, eval_env, eval_episodes)
    exp_config.actor_config = actor_config
    exp_config.critic_config = critic_config
    exp_config.seed = seed
    exp_config.model_cls = PPOClip
    return exp_config