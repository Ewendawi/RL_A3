
import time
from common.expConfig import join_processes, clean_processes, run_experiment
from common.plot import plot_with_file, write_dirs_to_file
from common.AbstractPGAlgorithm import render_env, train_model
from AC import exp_config_for_reinforce, REINFORCE

# experiment for reinforce
def experiment_reinforce():
    standardize_reward = False
    repeat = 1
    time_steps = 1000 * 1000
    batch_size = 0

    env_name = "CartPole-v1"
    actor_lr = 1e-3
    critic_lr = 1e-2

    env_name = "MountainCar-v0"
    batch_size = 64
    actor_lr = 1e-3
    critic_lr = 1e-3

    env_name = "LunarLander-v2"
    batch_size = 64
    actor_lr = 1e-3
    critic_lr = 5e-3

    # env_name = "Pendulum-v1"
    # standardize_reward = True
    # reward_uppper_bound = 0.0
    # reward_low_bound = -16.0
    # actor_lr = 1e-4
    # critic_lr = 1e-3

    # env_name = "Ant-v4"
    # standardize_reward = False
    # actor_lr = 1e-5
    # critic_lr = 1e-4 

    exp_name = f"reinforce-{env_name}-r{repeat}-t{time_steps}"

    exp_config = exp_config_for_reinforce(exp_name=exp_name, env_name=env_name, repeat=repeat, timesteps=time_steps)
    exp_config.critic_config.use_base_line = False
    exp_config.critic_config.use_gae = False
    exp_config.critic_config.batch_size = batch_size
    exp_config.actor_config.batch_size = batch_size
    exp_config.actor_config.learning_rate = actor_lr
    exp_config.critic_config.learning_rate = critic_lr
    exp_config.critic_config.standardize_reward = standardize_reward
    if standardize_reward:
        exp_config.critic_config.reward_uppper_bound = reward_uppper_bound
        exp_config.critic_config.reward_low_bound = reward_low_bound
    
    exp_config.tensorboard_dir = f"{exp_name}"

    model = REINFORCE(exp_config.actor_config, exp_config.critic_config)

    result = train_model(env=exp_config.env, model=model, time_steps=exp_config.timesteps, eval_interval=exp_config.eval_interval, eval_env=exp_config.eval_env, eval_episodes=exp_config.eval_episodes, seed=exp_config.seed, tensorboard_dir=exp_config.tensorboard_dir)

    now = time.strftime("%m%d_%H_%M_%S", time.localtime(time.time()))
    model_name = f"reinforce_{env_name}_r{repeat}_t{time_steps}_{now}"
    model_path = model.save(model_name)
    render_env(model_path)

'''
1. learning rate
2. batch size
3. entropy
4. baseline
'''
    
def experiment_learing_rate(multi_process=False, device_name="cpu", test_in_tensorboard=False):
    dirs = []
    run_multi_process = multi_process 
    repeat = 10
    time_steps = 1000 * 1000

    if test_in_tensorboard:
        run_multi_process = False
        repeat = 1
        time_steps = 1000 * 400

    env_name = "LunarLander-v2"

    lrs = [10e-4, 5e-4, 1e-4]

    exp_name = f"reinforce-{env_name}-lr-r{repeat}-t{time_steps}"
    for lr in lrs:
        exp_config = exp_config_for_reinforce(exp_name=exp_name, env_name=env_name, repeat=repeat, timesteps=time_steps, device_name=device_name)
        exp_config.critic_config.use_base_line = False
        exp_config.critic_config.use_gae = False
        exp_config.actor_config.learning_rate = lr

        exp_config.update_dir_name(str(lr))

        if test_in_tensorboard:
            exp_config.tensorboard_dir = f"{exp_name}-{lr}"

        run_experiment(exp_config=exp_config, multi_process=run_multi_process)
        dirs.append(exp_config.dir_name)

    if run_multi_process:
        join_processes()
        clean_processes()
    
    if test_in_tensorboard:
        return

    exp_labels = [str(lr) for lr in lrs]
    file = write_dirs_to_file(dirs=dirs, labels=exp_labels, file_name=exp_name)

    names = [f"{exp_name}_train", f"{exp_name}_eval"]
    plot_with_file(file, names=names)

def experiment_batch_size(multi_process=False, device_name="cpu", test_in_tensorboard=False):
    dirs = []
    run_multi_process = multi_process 
    repeat = 10
    time_steps = 1000 * 1000

    if test_in_tensorboard:
        run_multi_process = False
        repeat = 1
        time_steps = 1000 * 400

    env_name = "LunarLander-v2"
    learning_rate = 1e-3
    batch_sizes = [32, 64, 128]

    exp_name = f"reinforce-{env_name}-bs-r{repeat}-t{time_steps}"
    for batch_size in batch_sizes:
        exp_config = exp_config_for_reinforce(exp_name=exp_name, env_name=env_name, repeat=repeat, timesteps=time_steps, device_name=device_name)
        exp_config.critic_config.use_base_line = False
        exp_config.critic_config.use_gae = False
        exp_config.actor_config.learning_rate = learning_rate
        exp_config.actor_config.batch_size = batch_size

        exp_config.update_dir_name(str(batch_size))

        if test_in_tensorboard :
            exp_config.tensorboard_dir = f"{exp_name}-bs{batch_size}"

        run_experiment(exp_config=exp_config, multi_process=run_multi_process)
        dirs.append(exp_config.dir_name)

    if run_multi_process:
        join_processes()
        clean_processes()

    if test_in_tensorboard:
        return

    exp_labels = [str(batch_size) for batch_size in batch_sizes]
    file = write_dirs_to_file(dirs=dirs, labels=exp_labels, file_name=exp_name)

    names = [f"{exp_name}_train", f"{exp_name}_eval"]
    plot_with_file(file, names=names)

# experiment_batch_size(test_in_tensorboard=True)

def experiment_entropy(multi_process=False, device_name="cpu", test_in_tensorboard=False):
    dirs = []
    run_multi_process = multi_process 
    repeat = 10
    time_steps = 1000 * 1000

    if test_in_tensorboard:
        run_multi_process = False
        repeat = 1
        time_steps = 1000 * 400

    env_name = "LunarLander-v2"
    learning_rate = 1e-3

    entropies = [0, 0.1, 0.3]

    exp_name = f"reinforce-{env_name}-en-r{repeat}-t{time_steps}"
    for entropy in entropies:
        exp_config = exp_config_for_reinforce(exp_name=exp_name, env_name=env_name, repeat=repeat, timesteps=time_steps, device_name=device_name)
        exp_config.critic_config.use_base_line = False
        exp_config.critic_config.use_gae = False
        exp_config.actor_config.learning_rate = learning_rate
        exp_config.actor_config.entropy_weight = entropy

        exp_config.update_dir_name(str(entropy))

        if test_in_tensorboard:
            exp_config.tensorboard_dir = f"{exp_name}-en{entropy}"

        run_experiment(exp_config=exp_config, multi_process=run_multi_process)
        dirs.append(exp_config.dir_name)

    if run_multi_process:
        join_processes()
        clean_processes()

    if test_in_tensorboard:
        return

    exp_labels = [str(entropy) for entropy in entropies]
    file = write_dirs_to_file(dirs=dirs, labels=exp_labels, file_name=exp_name)

    names = [f"{exp_name}_train", f"{exp_name}_eval"]
    plot_with_file(file, names=names)

# experiment_entropy(test_in_tensorboard=True)

def experiment_baseline(multi_process=False, device_name="cpu", test_in_tensorboard=False):
    dirs = []
    run_multi_process = multi_process
    repeat = 10
    time_steps = 1000 * 1000

    if test_in_tensorboard:
        run_multi_process = False
        repeat = 1
        time_steps = 1000 * 400

    env_name = "LunarLander-v2"
    actor_lr = 1e-3
    critic_lr = 5e-3

    exp_name = f"reinforce-{env_name}-bl-r{repeat}-t{time_steps}"

    exp_config = exp_config_for_reinforce(exp_name=exp_name, env_name=env_name, repeat=repeat, timesteps=time_steps, device_name=device_name)
    exp_config.critic_config.use_base_line = False
    exp_config.critic_config.use_gae = False
    exp_config.actor_config.learning_rate = actor_lr 
    exp_config.update_dir_name("navie")
    if test_in_tensorboard :
        exp_config.tensorboard_dir = f"{exp_name}-bl-navie"
    run_experiment(exp_config=exp_config, multi_process=run_multi_process)
    dirs.append(exp_config.dir_name)

    # baseline
    exp_config = exp_config_for_reinforce(exp_name=exp_name, env_name=env_name, repeat=repeat, timesteps=time_steps, device_name=device_name)
    exp_config.critic_config.use_base_line = True
    exp_config.critic_config.use_gae = False
    exp_config.actor_config.learning_rate = actor_lr
    exp_config.critic_config.learning_rate = critic_lr
    exp_config.update_dir_name("baseline")
    if test_in_tensorboard :
        exp_config.tensorboard_dir = f"{exp_name}-bl-baseline"
    run_experiment(exp_config=exp_config, multi_process=run_multi_process)
    dirs.append(exp_config.dir_name)

    # gae
    exp_config = exp_config_for_reinforce(exp_name=exp_name, env_name=env_name, repeat=repeat, timesteps=time_steps, device_name=device_name)
    exp_config.critic_config.use_base_line = False
    exp_config.critic_config.use_gae = True
    exp_config.actor_config.learning_rate = actor_lr
    exp_config.update_dir_name("gae")
    if test_in_tensorboard :
        exp_config.tensorboard_dir = f"{exp_name}-bl-gae"
    run_experiment(exp_config=exp_config, multi_process=run_multi_process)
    dirs.append(exp_config.dir_name)

    if run_multi_process:
        join_processes()
        clean_processes()

    if test_in_tensorboard:
        return

    exp_labels = ["navie", "baseline", "gae"]
    file = write_dirs_to_file(dirs=dirs, labels=exp_labels, file_name=exp_name)

    names = [f"{exp_name}_train", f"{exp_name}_eval"]
    plot_with_file(file, names=names)

# experiment_baseline(test_in_tensorboard=True)