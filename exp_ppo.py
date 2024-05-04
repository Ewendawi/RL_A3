
from common.expConfig import join_processes, clean_processes, run_experiment
from common.plot import plot_with_file, write_dirs_to_file
from DDPG import DDPG, exp_config_for_DDPG
from PPOClip import PPOClip, exp_config_for_PPOClip


def experiment_DDPG():
    dirs = []
    run_multi_process = USE_MULTI_PROCESS and False
    repeat = 1
    time_steps = 20000
    batch_size = 64

    # env_name = "MountainCarContinuous-v0"
    env_name = "Pendulum-v1"
    env_name = "Ant-v4"
    exp_name = f"DDPG-r{repeat}-t{time_steps}-bs{batch_size}"

    exp_config = exp_config_for_DDPG(exp_name=exp_name, env_name=env_name, repeat=repeat, timesteps=time_steps, device_name=DEVICE_NAME)
    exp_config.critic_config.batch_size = batch_size
    exp_config.actor_config.batch_size = batch_size
    exp_config.update_dir_name("DDPG")

    model = DDPG(exp_config.actor_config, exp_config.critic_config)
    run_experiment(model, exp_config=exp_config, multi_process=run_multi_process)
    dirs.append(exp_config.dir_name)

    if run_multi_process:
        join_processes()
        clean_processes()

    exp_labels = []

    exp_labels = ["DDPG"]
    file = write_dirs_to_file(dirs=dirs, labels=exp_labels, file_name=exp_name)

    names = [f"{exp_name}_train", f"{exp_name}_eval"]
    plot_with_file(file, names=names)


def experiment_PPO():
    dirs = []
    run_multi_process = USE_MULTI_PROCESS and False
    repeat = 1
    batch_size = 64
    standardize_reward = False
    sample_size = 2048
    eval_interval = 3000

    env_name = "CartPole-v1"
    time_steps = 50000
    sample_size = 256
    actor_lr = 1e-3
    critic_lr = 3e-3

    env_name = "MountainCar-v0"
    batch_size = 64
    time_steps = 1000000
    eval_interval = 5000
    sample_size = 1000
    actor_lr = 1e-3
    critic_lr = 3e-2

    # env_name = "Pendulum-v1"
    # time_steps = 200000
    # sample_size = 256
    # standardize_reward = True
    # reward_uppper_bound = 0.0
    # reward_low_bound = -16.0
    # actor_lr = 1e-4
    # critic_lr = 1e-3

    # env_name = "Ant-v4"
    # time_steps = 200000
    # actor_lr = 1e-4
    # critic_lr = 1e-3

    env_name = "LunarLander-v2"
    time_steps = 1000000
    batch_size = 64
    actor_lr = 1e-4
    critic_lr = 3e-5
    sample_size = 2048 

    exp_name = f"PPO-{env_name}-r{repeat}-t{time_steps}"

    exp_config = exp_config_for_PPOClip(exp_name=exp_name, env_name=env_name, repeat=repeat, timesteps=time_steps, device_name=DEVICE_NAME)
    exp_config.actor_config.entropy_weight = 0.00
    # exp_config.actor_config.layers_dim = [64]
    # exp_config.critic_config.layers_dim = [64]
    exp_config.actor_config.batch_size = batch_size
    exp_config.critic_config.batch_size = batch_size
    
    exp_config.eval_interval = eval_interval
    exp_config.actor_config.learning_rate = actor_lr
    exp_config.critic_config.learning_rate = critic_lr
    exp_config.critic_config.max_episode_length = sample_size
    exp_config.critic_config.standardize_reward = standardize_reward
    if standardize_reward:
        exp_config.critic_config.reward_uppper_bound = reward_uppper_bound
        exp_config.critic_config.reward_low_bound = reward_low_bound
    exp_config.tensorboard_dir = f"{exp_name}"
    exp_config.update_dir_name("PPO")

    model = PPOClip(exp_config.actor_config, exp_config.critic_config)
    run_experiment(model, exp_config=exp_config, multi_process=run_multi_process)
    dirs.append(exp_config.dir_name)
    # model.save("ppo_model")

    if run_multi_process:
        join_processes()
        clean_processes()

    exp_labels = []

    exp_labels = ["ppo"]
    file = write_dirs_to_file(dirs=dirs, labels=exp_labels, file_name=exp_name)

    names = [f"{exp_name}_train", f"{exp_name}_eval"]
    plot_with_file(file, names=names)


def experiment_ppo_kl(multi_process=False, device_name="cpu"):
    dirs = []
    run_multi_process = multi_process
    repeat = 10

    time_steps = 1000 * 1000

    test_in_tensorboard = False
    if test_in_tensorboard:
        run_multi_process = False
        repeat = 1
        time_steps = 1000 * 400

    env_name = "LunarLander-v2"
    eval_interval = 3000
    time_steps = 1000000
    batch_size = 64
    actor_lr = 1e-4
    critic_lr = 3e-5
    sample_size = 2048 

    exp_name = f"PPO-{env_name}-kl-r{repeat}-t{time_steps}"

    # without kl clip
    exp_config = exp_config_for_PPOClip(exp_name=exp_name, env_name=env_name, repeat=repeat, timesteps=time_steps, device_name=device_name)
    exp_config.actor_config.entropy_weight = 0.00
    exp_config.actor_config.batch_size = batch_size
    exp_config.critic_config.batch_size = batch_size
    exp_config.eval_interval = eval_interval
    exp_config.actor_config.learning_rate = actor_lr
    exp_config.critic_config.learning_rate = critic_lr
    exp_config.critic_config.max_episode_length = sample_size
    if test_in_tensorboard:
        exp_config.tensorboard_dir = f"{exp_name}-no-kl"
    exp_config.update_dir_name("PPO")
    model = PPOClip(exp_config.actor_config, exp_config.critic_config)
    run_experiment(model, exp_config=exp_config, multi_process=run_multi_process)
    dirs.append(exp_config.dir_name)

    # with kl clip 
    exp_config = exp_config_for_PPOClip(exp_name=exp_name, env_name=env_name, repeat=repeat, timesteps=time_steps, device_name=device_name)
    exp_config.actor_config.entropy_weight = 0.00
    exp_config.actor_config.batch_size = batch_size
    exp_config.critic_config.batch_size = batch_size
    exp_config.eval_interval = eval_interval
    exp_config.actor_config.learning_rate = actor_lr
    exp_config.critic_config.learning_rate = critic_lr
    exp_config.critic_config.max_episode_length = sample_size
    exp_config.critic_config.target_kl = 0.007
    if test_in_tensorboard:
        exp_config.tensorboard_dir = f"{exp_name}-kl"
    exp_config.update_dir_name("PPO")
    model = PPOClip(exp_config.actor_config, exp_config.critic_config)
    run_experiment(model, exp_config=exp_config, multi_process=run_multi_process)
    dirs.append(exp_config.dir_name)

    if run_multi_process:
        join_processes()
        clean_processes()

    if test_in_tensorboard:
        return

    exp_labels = ["ppo"]
    file = write_dirs_to_file(dirs=dirs, labels=exp_labels, file_name=exp_name)

    names = [f"{exp_name}_train", f"{exp_name}_eval"]
    plot_with_file(file, names=names)
