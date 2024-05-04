
from common.expConfig import join_processes, clean_processes, run_experiment
from common.plot import write_dirs_to_file, plot_file_with_keys
from DDPG import exp_config_for_DDPG
from PPOClip import exp_config_for_PPOClip


def experiment_DDPG():
    repeat = 1
    time_steps = 20000
    batch_size = 64

    # env_name = "MountainCarContinuous-v0"
    env_name = "Pendulum-v1"
    env_name = "Ant-v4"
    exp_name = f"DDPG-{env_name}-t{time_steps}"

    exp_config = exp_config_for_DDPG(exp_name=exp_name, env_name=env_name, repeat=repeat, timesteps=time_steps, device_name="cpu")
    exp_config.critic_config.batch_size = batch_size
    exp_config.actor_config.batch_size = batch_size
    exp_config.tensorboard_dir = f"{exp_name}"

    run_experiment(exp_config=exp_config)


def experiment_PPO():
    run_multi_process = False
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
    time_steps = 1000 * 1000
    batch_size = 32
    actor_lr = 1e-4
    critic_lr = 3e-5
    sample_size = 2048 
    ppo_repeat = 10
    kl_target = 0.000
    kl_target = None
    use_total_loss = False
    value_loss_weight = 0.3

    exp_name = f"PPO-{env_name}-r{repeat}-t{time_steps}"

    exp_config = exp_config_for_PPOClip(exp_name=exp_name, env_name=env_name, repeat=repeat, timesteps=time_steps, device_name="cpu")
    exp_config.actor_config.entropy_weight = 0.00
    exp_config.actor_config.batch_size = batch_size
    exp_config.critic_config.batch_size = batch_size
    exp_config.critic_config.repeat = ppo_repeat
    exp_config.critic_config.target_kl = kl_target
    exp_config.critic_config.use_total_loss = use_total_loss
    exp_config.critic_config.value_loss_weight = value_loss_weight

    exp_config.eval_interval = eval_interval
    exp_config.actor_config.learning_rate = actor_lr
    exp_config.critic_config.learning_rate = critic_lr
    exp_config.critic_config.max_episode_length = sample_size
    exp_config.critic_config.standardize_reward = standardize_reward
    if standardize_reward:
        exp_config.critic_config.reward_uppper_bound = reward_uppper_bound
        exp_config.critic_config.reward_low_bound = reward_low_bound
    exp_config.tensorboard_dir = f"{exp_name}"

    run_experiment(exp_config=exp_config, multi_process=run_multi_process)

# experiment_PPO()

def experiment_ppo_lr(multi_process=False, device_name="cpu", test_in_tensorboard=False):
    dirs = []
    run_multi_process = multi_process
    repeat = 5

    time_steps = 1000 * 1000

    if test_in_tensorboard:
        run_multi_process = False
        repeat = 1
        time_steps = 1000 * 400

    env_name = "LunarLander-v2"
    batch_size = 64
    critic_lr = 3e-5
    sample_size = 2048 

    exp_name = f"PPO-{env_name}-lr-r{repeat}-t{time_steps}"

    lrs = [8e-4, 1e-4, 5e-5]

    for lr in lrs:
        exp_config = exp_config_for_PPOClip(exp_name=exp_name, env_name=env_name, repeat=repeat, timesteps=time_steps, device_name=device_name)

        exp_config.actor_config.entropy_weight = 0.00
        exp_config.actor_config.batch_size = batch_size
        exp_config.actor_config.learning_rate = lr 

        exp_config.critic_config.batch_size = batch_size
        exp_config.critic_config.learning_rate = critic_lr
        exp_config.critic_config.max_episode_length = sample_size

        if test_in_tensorboard:
            exp_config.tensorboard_dir = f"{exp_name}-{lr}"
        exp_config.update_dir_name(str(lr))
        run_experiment(exp_config=exp_config, multi_process=run_multi_process)
        dirs.append(exp_config.dir_name)

    if run_multi_process:
        join_processes()
        clean_processes()

    if test_in_tensorboard:
        return

    exp_labels = [str(lr) for lr in lrs]
    file = write_dirs_to_file(dirs=dirs, labels=exp_labels, file_name=exp_name)

    keys = {
        "eval": ["eval", f"{exp_name}_eval"],
        "critic_loss": ["critic_loss", f"{exp_name}_critic_loss"],
        "approx_kl_div": ["kl_div", f"{exp_name}_approx_kl_div"],
    }
    plot_file_with_keys(file, keys)


def experiment_ppo_kl(multi_process=False, device_name="cpu", test_in_tensorboard=False):
    dirs = []
    run_multi_process = multi_process
    repeat = 5

    time_steps = 1000 * 1000

    if test_in_tensorboard:
        run_multi_process = False
        repeat = 1
        time_steps = 1000 * 400

    env_name = "LunarLander-v2"
    batch_size = 64
    actor_lr = 8e-4
    critic_lr = 3e-5
    sample_size = 2048 

    exp_name = f"PPO-{env_name}-kl-r{repeat}-t{time_steps}"

    kl_clips = [0.01, 0.0085, 0.007, None]

    for kl_clip in kl_clips:
        exp_config = exp_config_for_PPOClip(exp_name=exp_name, env_name=env_name, repeat=repeat, timesteps=time_steps, device_name=device_name)

        exp_config.actor_config.entropy_weight = 0.00
        exp_config.actor_config.batch_size = batch_size
        exp_config.actor_config.learning_rate = actor_lr

        exp_config.critic_config.batch_size = batch_size
        exp_config.critic_config.learning_rate = critic_lr
        exp_config.critic_config.max_episode_length = sample_size
        exp_config.critic_config.target_kl = kl_clip

        if test_in_tensorboard:
            exp_config.tensorboard_dir = f"{exp_name}-{kl_clip}"
        exp_config.update_dir_name(str(kl_clip))
        run_experiment(exp_config=exp_config, multi_process=run_multi_process)
        dirs.append(exp_config.dir_name)

    if run_multi_process:
        join_processes()
        clean_processes()

    if test_in_tensorboard:
        return

    exp_labels = [str(kl_clip) for kl_clip in kl_clips]
    file = write_dirs_to_file(dirs=dirs, labels=exp_labels, file_name=exp_name)

    keys = {
        "eval": ["eval", f"{exp_name}_eval"],
        "critic_loss": ["critic_loss", f"{exp_name}_critic_loss"],
        "approx_kl_div": ["kl_div", f"{exp_name}_approx_kl_div"],
    }
    plot_file_with_keys(file, keys)  


def experiment_ppo_repeat(multi_process=False, device_name="cpu", test_in_tensorboard=False):
    dirs = []
    run_multi_process = multi_process
    repeat = 5
    time_steps = 1000 * 1000

    if test_in_tensorboard:
        run_multi_process = False
        repeat = 1
        time_steps = 1000 * 400

    env_name = "LunarLander-v2"
    batch_size = 64
    actor_lr = 8e-4
    critic_lr = 3e-5
    sample_size = 2048 

    exp_name = f"PPO-{env_name}-repeat-r{repeat}-t{time_steps}"

    ppo_repeats = [3, 6, 10]
    for ppo_repeat in ppo_repeats:
        exp_config = exp_config_for_PPOClip(exp_name=exp_name, env_name=env_name, repeat=repeat, timesteps=time_steps, device_name=device_name)

        exp_config.actor_config.entropy_weight = 0.00
        exp_config.actor_config.batch_size = batch_size
        exp_config.actor_config.learning_rate = actor_lr

        exp_config.critic_config.batch_size = batch_size
        exp_config.critic_config.learning_rate = critic_lr
        exp_config.critic_config.max_episode_length = sample_size

        exp_config.critic_config.repeat = ppo_repeat

        if test_in_tensorboard:
            exp_config.tensorboard_dir = f"{exp_name}-{repeat}"
        exp_config.update_dir_name(str(repeat))
        run_experiment(exp_config=exp_config, multi_process=run_multi_process)
        dirs.append(exp_config.dir_name)

    if run_multi_process:
        join_processes()
        clean_processes()

    if test_in_tensorboard:
        return

    exp_labels = [str(repeat) for repeat in ppo_repeats]
    file = write_dirs_to_file(dirs=dirs, labels=exp_labels, file_name=exp_name)

    keys = {
        "eval": ["eval", f"{exp_name}_eval"],
        "critic_loss": ["critic_loss", f"{exp_name}_critic_loss"],
        "approx_kl_div": ["kl_div", f"{exp_name}_approx_kl_div"],
    }
    plot_file_with_keys(file, keys)
