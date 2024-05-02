
import gymnasium as gym
import numpy as np
import torch
from common.networks import ContinuousPolicyNetwork, DiscretePolicyNetwork, ValueNetwork
from common.helper import SampleBuffer, batch_index_generator
from common.AbstractPGAlgorithm import CriticConfig, ActorConfig, AbstractActor, AbstractCritic, AbstractPGAlgorithm, save_model
from common.expConfig import ExpConfig

# torch.autograd.set_detect_anomaly(True)

class StochasticActor(AbstractActor):
    def __init__(self, config):
        super(StochasticActor, self).__init__(config)
        if self.config.continuous_action_space:
            action_bound = torch.tensor(config.action_bound, dtype=torch.float32).to(config.device)
            self.policy_net = ContinuousPolicyNetwork(config.input_size, config.layers_dim, config.output_size, action_bound).to(config.device)
        else:
            self.policy_net = DiscretePolicyNetwork(config.input_size, config.layers_dim, config.output_size).to(config.device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)

    def save(self, name):
        save_model(self.policy_net, name)

    def action_distribution(self, state):
        if self.config.continuous_action_space:
            mu, std = self.policy_net(state)
            action_dist = torch.distributions.Normal(mu, std)
        else:
            probs = self.policy_net(state)
            action_dist = torch.distributions.Categorical(probs)
        return action_dist

    def log_probs_with_dists(self, action_dist, actions):
        if self.config.continuous_action_space:
            log_probs = action_dist.log_prob(actions)
        else:
            actions = actions.view(1,-1)
            log_probs = action_dist.log_prob(actions).view(-1,1)
        return log_probs

    def entropy_with_dists(self, action_dist):
        entropy = action_dist.entropy()
        if not self.config.continuous_action_space:
            entropy = entropy.view(-1,1)
        return entropy

    def take_action(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).view(1,-1).to(self.config.device)
            action_dist = self.action_distribution(state)
            action = action_dist.sample().cpu()
            if self.config.continuous_action_space:
                # shift action to the action bound
                action = action + self.config.action_low_bound + self.config.action_bound
                if self.config.gsde_noise_scale > 0:
                    action = action + torch.randn_like(action) * self.config.gsde_noise_scale
                # low_bound = torch.tensor(self.config.action_low_bound, dtype=torch.float32).to(self.config.device)
                # upper_bound = torch.tensor(self.config.action_uppper_bound, dtype=torch.float32).to(self.config.device)
                # action = torch.clamp(action, low_bound, upper_bound)
        return action.flatten().tolist()

    def update(self, samples, psi_values):
        states, actions, _, _, _= samples

        batch_size = len(psi_values) if len(psi_values) < self.config.batch_size or self.config.batch_size == 0 else self.config.batch_size
        total_loss = []
        for i,j in batch_index_generator(len(psi_values), batch_size):
            dists = self.action_distribution(states[i:j])
            log_probs = self.log_probs_with_dists(dists, actions[i:j])
            entropy = self.entropy_with_dists(dists)
            loss = -torch.mean(log_probs * psi_values[i:j] - self.config.entropy_weight * entropy)
            total_loss.append(loss.detach().cpu())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return np.mean(total_loss)

def GAE_estimate(rewards, values, dones, gamma, lam, next_value=0):
    T = len(rewards)
    advantages = torch.zeros(T).to(values.device)
    gae = 0
    next_value = next_value
    for t in reversed(range(T)):
        reward = rewards[t]
        value = values[t]
        done = dones[t]
        # td_error = r_t + gamma * v_{t+1} * (1-done_{t+1}) - v_t
        delta = reward + gamma * next_value * (1 - done) - value
        # gae_t = delta + gamma * lambda * gae_{t+1}
        gae = delta + gamma * lam * gae * (1 - done)
        advantages[t] = gae
        next_value = value
    return advantages

    
class NormalCritic(AbstractCritic):
    def __init__(self, config):
        super(NormalCritic, self).__init__(config)
        if self.should_use_value_net():
            self.value_net = ValueNetwork(config.input_size, config.layers_dim).to(config.device)
            self.value_net_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=config.learning_rate)

    def should_use_value_net(self):
        return self.config.use_base_line or self.config.use_gae or self.config.n_steps > 0

    def returns_of_episode(self, rewards, states, states_, dones):
        if self.config.n_steps == 0:
            # MC
            # G_t = sum_{i=0}^{n-1} gamma^i * r_{t+i} , n = length of episode
            T = len(rewards)
            returns = np.zeros(T)
            G = 0
            for t in reversed(range(T)):
                reward = rewards[t]
                G = self.config.gamma * G + reward
                returns[t] = G
            returns = torch.tensor(returns, dtype=torch.float32).view(-1,1).to(self.config.device)
            return returns    
        else:
            state_values = self.value_net(states).detach()
            # n-step
            # G = sum_{i=0}^{n-1} gamma^i * r_{t+i} + gamma^n * V(s_{t+n}) , n = n_steps
            G_list = []
            for t in range(len(rewards)):
                done = dones[t]
                G = 0
                actual_steps = min(self.config.n_steps, len(rewards)-t)
                for j in range(t, t+actual_steps):
                    G += self.config.gamma**(j-t) * rewards[j]
                if not done and t + actual_steps != len(rewards):
                    next_state_value = state_values[t+actual_steps].item()
                    G += self.config.gamma**self.config.n_steps * next_state_value
                G_list.append(G)
            G_list = torch.tensor(G_list, dtype=torch.float32).view(-1,1).to(self.config.device)
            return G_list

    def calculate_psi_value(self, returns, rewards, state_values, dones):
        if self.config.use_gae:
            advantages = GAE_estimate(rewards, state_values, dones, self.config.gamma, self.config.gae_lambda)
            psi_values = advantages.view(-1,1)
        elif self.config.use_base_line:
            psi_values = returns - state_values
        else:
            psi_values = returns
        return psi_values.detach()

    def update(self, samples):
        states, _, rewards, states_, dones = samples

        returns = self.returns_of_episode(rewards, states, states_, dones)

        # calculate value network loss
        total_loss = []
        if self.should_use_value_net():
            state_values = []
            batch_size = len(states) if len(states) < self.config.batch_size or self.config.batch_size == 0 else self.config.batch_size
            for i,j in batch_index_generator(len(states), batch_size):
                b_state_values = self.value_net(states[i:j])
                b_returns = returns[i:j]
                value_loss = torch.nn.functional.mse_loss(b_state_values, b_returns)
                self.value_net_optimizer.zero_grad() 
                value_loss.backward()
                self.value_net_optimizer.step()

                state_values.append(b_state_values)
                total_loss.append(value_loss.detach().cpu())
            state_values = torch.cat(state_values, dim=0)
        else:
            state_values = torch.zeros_like(returns)

        psi_values = self.calculate_psi_value(returns, rewards, state_values, dones)

        return np.mean(total_loss), psi_values 

class OnPolicyPGAlgorithm(AbstractPGAlgorithm):
    def __init__(self, actor_config: ActorConfig, critic_config: CriticConfig):
        super(OnPolicyPGAlgorithm, self).__init__(actor_config, critic_config)
        self.actor_config = actor_config
        self.critic_config = critic_config
        self.actor: AbstractActor
        self.critic: AbstractCritic
        self.sample_buffer = SampleBuffer()

    def save(self, name):
        self.actor.save(name)

    def store_transition(self, state, action, reward, next_state, done, ):
        self.sample_buffer.store_transition(state, action, reward, next_state, done)

    def should_update(self, done, truncated):
        if done or truncated:
            return True
        if self.sample_buffer.size() >= self.critic_config.max_episode_length:
            return True
        return False

    def take_action(self, state):
        return self.actor.take_action(state)

    def update(self):
        samples = self.sample_buffer.get_all_samples()
        states, actions, rewards, states_, dones = samples
        states = torch.tensor(states, dtype=torch.float32).to(self.actor_config.device)
        actions = torch.tensor(actions).view(-1,1).to(self.actor_config.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).view(-1,1).to(self.actor_config.device)
        states_ = torch.tensor(states_, dtype=torch.float32).to(self.actor_config.device)
        dones = torch.tensor(dones, dtype=torch.float32).view(-1,1).to(self.actor_config.device)
        samples = (states, actions, rewards, states_, dones)

        if self.critic.config.standardize_reward:
            mean_reward = (self.critic.config.reward_uppper_bound - self.critic.config.reward_low_bound)/2
            rewards = (rewards + mean_reward) / mean_reward
        
        critic_loss, psi_values = self.critic.update(samples)
        actor_loss = self.actor.update(samples, psi_values)

        self.sample_buffer.reset()
        return critic_loss, actor_loss

def exp_config_for_AC(exp_name="AC", env_name="", repeat=1, timesteps=20000, device_name="cpu"):
    seed = 123

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

    critic_config = CriticConfig(device_name=device_name)
    critic_config.input_size = env.observation_space.shape[0]
    critic_config.n_steps = 1

    eval_interval = 3000
    eval_env = gym.make(env_name)
    eval_episodes = 10

    exp_config = ExpConfig(exp_name, repeat, env, timesteps, eval_interval, eval_env, eval_episodes)
    exp_config.actor_config = actor_config
    exp_config.critic_config = critic_config
    exp_config.seed = seed
    return exp_config

class ActorCritic(OnPolicyPGAlgorithm):
    def __init__(self, actor_config, critic_config):
        super(ActorCritic, self).__init__(actor_config, critic_config)
        self.actor = StochasticActor(actor_config)
        self.critic = NormalCritic(critic_config)


def exp_config_for_reinforce(exp_name="REINFORCE", env_name="", repeat=1, timesteps=20000, device_name="cpu"):
    seed = 123

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

    critic_config = CriticConfig(device_name=device_name)
    critic_config.input_size = env.observation_space.shape[0]
    critic_config.n_steps = 0

    eval_interval = 5000
    eval_env = gym.make(env_name)
    eval_episodes = 10

    exp_config = ExpConfig(exp_name, repeat, env, timesteps, eval_interval, eval_env, eval_episodes)
    exp_config.actor_config = actor_config
    exp_config.critic_config = critic_config
    exp_config.seed = seed
    return exp_config

class REINFORCE(OnPolicyPGAlgorithm):
    def __init__(self, actor_config, critic_config):
        super(REINFORCE, self).__init__(actor_config, critic_config)
        self.actor = StochasticActor(actor_config)
        self.critic = NormalCritic(critic_config)


