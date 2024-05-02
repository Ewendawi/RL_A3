
import torch
import gymnasium as gym
from common.networks import DeterministicContinuousPolicyNetwork, QNetworkContinnous
from common.helper import replayBuffer
from common.AbstractPGAlgorithm import AbstractActor, AbstractCritic, ActorConfig, CriticConfig, AbstractPGAlgorithm, save_model
from common.expConfig import ExpConfig

class DDPGActorConfig(ActorConfig):
    def __init__(self, device_name="cpu"):
        super(DDPGActorConfig, self).__init__(device_name)
        self.continuous_action_space = True
        # self.layers_dim = [128, 128]
        self.action_bound = 1
        self.noise_std = 0.1 
        self.noise_clip = 0.2
        self.update_factor = 0.005
        self.learning_rate = 0.0003

class DDPGActor(AbstractActor):
    def __init__(self, config: DDPGActorConfig):
        super(DDPGActor, self).__init__(config)
        self.policy_net = DeterministicContinuousPolicyNetwork(config.input_size, config.layers_dim, config.output_size, config.action_bound).to(config.device)
        self.policy_net_target = DeterministicContinuousPolicyNetwork(config.input_size, config.layers_dim, config.output_size, config.action_bound).to(config.device)
        self.policy_net_target.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)
        self.config = config
    
    def save(self, name):
        save_model(self.policy_net, name)

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).view(1,-1).to(self.config.device)
        action = self.policy_net(state)
        noise = torch.normal(0, self.config.noise_std, (1,)).to(self.config.device)
        noise = torch.clamp(noise, -self.config.noise_clip, self.config.noise_clip)
        action = action + noise
        # action = torch.clamp(action, -self.config.action_bound, self.config.action_bound)
        return action.detach().cpu().numpy().flatten()

    def update(self, samples, psi_values):
        for target_param, param in zip(self.policy_net_target.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.config.update_factor * target_param.data + (1 - self.config.update_factor) * param.data)


class DDPGCriticConfig(CriticConfig):
    def __init__(self, device_name="cpu"):
        super(DDPGCriticConfig, self).__init__(device_name)
        self.action_space_dim = 1
        self.state_dim = [64]
        self.action_dim = [32]
        self.layers_dim = [64]
        self.n_steps = 1
        self.target_update_factor = 0.005
        self.replay_buffer_size = 10000
        self.replay_buffer_warmup_size = 1000
        self.learning_rate = 0.003
        
class DDPGCritic(AbstractCritic):
    def __init__(self, config:DDPGCriticConfig):
        super(DDPGCritic, self).__init__(config)
        self.config = config
        self.buffer = replayBuffer(config.replay_buffer_size, config.n_steps, config.gamma, config.replay_buffer_warmup_size)
        self.value_net = QNetworkContinnous(config.input_size, config.state_dim, config.action_space_dim, config.action_dim, config.layers_dim).to(config.device)
        self.value_net_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=config.learning_rate)
        self.target_value_net = QNetworkContinnous(config.input_size, config.state_dim, config.action_space_dim, config.action_dim, config.layers_dim).to(config.device)
        self.target_value_net.load_state_dict(self.value_net.state_dict())

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.store_transition(state, action, reward, next_state, done)

    def update(self, samples):
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(self.config.target_update_factor * param.data + (1 - self.config.target_update_factor) * target_param.data) 


# tricks for DDPG
# - replay buffer
# - target policy network and target value network
# - clip noise
        
class DDPG(AbstractPGAlgorithm):
    def __init__(self, actor_config:DDPGActorConfig, critic_config:DDPGCriticConfig):
        super(DDPG, self).__init__(actor_config, critic_config)
        self.actor = DDPGActor(actor_config)
        self.critic = DDPGCritic(critic_config)
        self.sample_buffer = replayBuffer(critic_config.replay_buffer_size, critic_config.n_steps, critic_config.gamma, critic_config.replay_buffer_warmup_size)

    def take_action(self, state):
        return self.actor.take_action(state)

    def store_transition(self, state, action, reward, next_state, done):
        self.sample_buffer.store_transition(state, action, reward, next_state, done)

    def should_update(self, done, truncated):
        if self.sample_buffer.size() >= self.critic.config.replay_buffer_warmup_size:
            return True
        return False

    def update(self):
        samples = self.sample_buffer.sample_buffer(self.critic.config.batch_size)
        states, actions, rewards, states_, dones = samples
        states = torch.tensor(states, dtype=torch.float).to(self.critic.config.device)
        actions = torch.tensor(actions).to(self.critic.config.device)
        if len(actions.shape) == 1:
            actions = actions.view(-1,1)
        rewards = torch.tensor(rewards, dtype=torch.float).view(-1,1).to(self.critic.config.device)
        states_ = torch.tensor(states_, dtype=torch.float).to(self.critic.config.device)
        dones = torch.tensor(dones, dtype=torch.float).view(-1,1).to(self.critic.config.device)

        returns = self.critic.target_value_net(states_, self.actor.policy_net_target(states_))
        psi_values = rewards + self.critic.config.gamma * (1 - dones) * returns
        critic_loss = torch.nn.functional.mse_loss(self.critic.value_net(states, actions), psi_values).mean()
        self.critic.value_net_optimizer.zero_grad()
        critic_loss.backward()
        self.critic.value_net_optimizer.step()
        self.critic.update(samples)
        
        actor_loss = -self.critic.value_net(states, self.actor.policy_net(states)).mean()
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()
        self.actor.update(samples, psi_values)

        return critic_loss.detach().item(), actor_loss.detach().item()

        
def exp_config_for_DDPG(exp_name="DDPG", env_name="", repeat=1, timesteps=20000, device_name="cpu"):
    seed = 123

    env = gym.make(env_name)
    
    actor_config = DDPGActorConfig(device_name=device_name)
    actor_config.input_size = env.observation_space.shape[0]
    actor_config.output_size = env.action_space.shape[0]
    actor_config.continuous_action_space = True 
    actor_config.action_bound = env.action_space.high[0]

    critic_config = DDPGCriticConfig(device_name=device_name)
    critic_config.input_size = env.observation_space.shape[0]
    critic_config.n_steps = 1
    critic_config.action_space_dim = env.action_space.shape[0]

    eval_interval = 1000
    eval_env = gym.make(env_name)
    eval_episodes = 10

    exp_config = ExpConfig(exp_name, repeat, env, timesteps, eval_interval, eval_env, eval_episodes)
    exp_config.actor_config = actor_config
    exp_config.critic_config = critic_config
    exp_config.seed = seed
    return exp_config