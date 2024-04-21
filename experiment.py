
from random import sample
import sys
import getopt
from common.expConfig import join_processes, clean_processes, run_experiment
from common.plot import plot_with_file, write_dirs_to_file, clear_files
from common.AbstractPGAlgorithm import render_env
from AC import ActorCritic, exp_config_for_AC, REINFORCE, exp_config_for_reinforce 
from DDPG import DDPG, exp_config_for_DDPG
from PPOClip import PPOClip, exp_config_for_PPOClip


USE_MULTI_PROCESS = True
DEVICE_NAME = "cpu"

# experiment for reinforce
def experiment_reinforce():
    dirs = []
    run_multi_process = USE_MULTI_PROCESS and False
    standardize_reward = False
    repeat = 1
    time_steps = 400000
    batch_size = 0

    env_name = "CartPole-v1"
    actor_lr = 1e-3
    critic_lr = 1e-2

    # env_name = "Pendulum-v1"
    # standardize_reward = True
    # reward_uppper_bound = 0.0
    # reward_low_bound = -16.0
    # actor_lr = 1e-4
    # critic_lr = 1e-3

    env_name = "Ant-v4"
    standardize_reward = False
    actor_lr = 1e-5
    critic_lr = 1e-4 

    exp_name = f"reinforce-r{repeat}-t{time_steps}-bs{batch_size}"

    exp_config = exp_config_for_reinforce(exp_name=exp_name, env_name=env_name, repeat=repeat, timesteps=time_steps, device_name=DEVICE_NAME)
    exp_config.critic_config.use_base_line = True
    exp_config.critic_config.use_gae = False
    exp_config.critic_config.batch_size = batch_size
    exp_config.actor_config.learning_rate = actor_lr
    exp_config.critic_config.learning_rate = critic_lr
    exp_config.critic_config.standardize_reward = standardize_reward
    if standardize_reward:
        exp_config.critic_config.reward_uppper_bound = reward_uppper_bound
        exp_config.critic_config.reward_low_bound = reward_low_bound
    
    exp_config.update_dir_name('re')
    model = REINFORCE(exp_config.actor_config, exp_config.critic_config)
    run_experiment(model, exp_config=exp_config, multi_process=run_multi_process)
    dirs.append(exp_config.dir_name)

    if run_multi_process:
        join_processes()
        clean_processes()

    exp_labels = []

    exp_labels = ["re"]
    file = write_dirs_to_file(dirs=dirs, labels=exp_labels, file_name=exp_name)

    names = [f"{exp_name}_train", f"{exp_name}_eval"]
    plot_with_file(file, names=names)
    
def experiment_AC():
    dirs = []
    run_multi_process = USE_MULTI_PROCESS and False
    repeat = 1
    standardize_reward = False
    time_steps = 200000
    batch_size = 0

    env_name = "CartPole-v1"
    actor_lr = 1e-3
    critic_lr = 1e-2

    env_name = "Pendulum-v1"
    standardize_reward = True
    reward_uppper_bound = 0.0
    reward_low_bound = -16.0
    actor_lr = 1e-4
    critic_lr = 1e-3

    # env_name = "Ant-v4"
    # standardize_reward = False
    # actor_lr = 1e-4
    # critic_lr = 1e-3 

    exp_name = f"AC-{env_name}-r{repeat}-t{time_steps}"
    model_name = f"AC_{env_name}_r{repeat}_t{time_steps}"

    exp_config = exp_config_for_AC(exp_name=exp_name, env_name=env_name, repeat=repeat, timesteps=time_steps, device_name=DEVICE_NAME)
    exp_config.actor_config.batch_size = batch_size
    exp_config.actor_config.learning_rate = actor_lr

    exp_config.critic_config.use_base_line = True
    exp_config.critic_config.use_gae = False
    exp_config.critic_config.n_steps = 5
    exp_config.critic_config.batch_size = batch_size
    exp_config.critic_config.learning_rate = critic_lr
    exp_config.critic_config.standardize_reward = standardize_reward
    if standardize_reward:
        exp_config.critic_config.reward_uppper_bound = reward_uppper_bound
        exp_config.critic_config.reward_low_bound = reward_low_bound
    
    exp_config.update_dir_name("AC")

    model = ActorCritic(exp_config.actor_config, exp_config.critic_config)
    run_experiment(model, exp_config=exp_config, multi_process=run_multi_process)
    model.save(model_name)

    dirs.append(exp_config.dir_name)

    if run_multi_process:
        join_processes()
        clean_processes()

    exp_labels = []

    exp_labels = ["AC"]
    file = write_dirs_to_file(dirs=dirs, labels=exp_labels, file_name=exp_name)

    names = [f"{exp_name}_train", f"{exp_name}_eval"]
    plot_with_file(file, names=names)


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
    batch_size = 0
    standardize_reward = False
    sample_size = 2048

    env_name = "CartPole-v1"
    time_steps = 50000
    sample_size = 256
    actor_lr = 1e-3
    critic_lr = 1e-2

    env_name = "Pendulum-v1"
    time_steps = 200000
    sample_size = 256
    standardize_reward = True
    reward_uppper_bound = 0.0
    reward_low_bound = -16.0
    actor_lr = 1e-4
    critic_lr = 1e-3

    # env_name = "Ant-v4"
    # time_steps = 200000
    # actor_lr = 1e-4
    # critic_lr = 1e-3

    exp_name = f"PPO-r{repeat}-t{time_steps}-bs{batch_size}"

    exp_config = exp_config_for_PPOClip(exp_name=exp_name, env_name=env_name, repeat=repeat, timesteps=time_steps, device_name=DEVICE_NAME)
    exp_config.actor_config.learning_rate = actor_lr
    exp_config.critic_config.learning_rate = critic_lr
    exp_config.critic_config.max_episode_length = sample_size
    exp_config.critic_config.standardize_reward = standardize_reward
    if standardize_reward:
        exp_config.critic_config.reward_uppper_bound = reward_uppper_bound
        exp_config.critic_config.reward_low_bound = reward_low_bound
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


if __name__ == "__main__":
    argv = sys.argv[1:]
    experiment = 2
    try:
        opts, args = getopt.getopt(argv,"he:", ["disable_multi_process", "device="])
    except getopt.GetoptError:
        print('python experiment.py -e 0 --device=cpu')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('python main.py -e 0')
            sys.exit()
        elif opt in ("-e"):
            experiment = int(arg)
        elif opt in ("--disable_multi_process"):
            USE_MULTI_PROCESS = False
        elif opt in ("--device"):
            DEVICE_NAME = arg

    # 0: reinforce
    # 1: AC
    # 2: DDPG
    # 3: PPO
    # 10: render

    if experiment == 0:
        print("Running experiment reinforce")
        experiment_reinforce()
    elif experiment == 1:
        print("Running experiment AC")
        experiment_AC()
    elif experiment == 2:
        print("Running experiment DDPG")
        experiment_DDPG()
    elif experiment == 3:
        print("Running experiment PPO")
        experiment_PPO()
    elif experiment == 10:
        path = "./models/AC_Pendulum-v1_True_r1_t200000_0412_23_45_31"
        render_env(path)
    
    # plot_with_file("DQN_rb_tn-r15-t20000-bs128-0330_16_36_30.npy")

    # clear the results you don't need.
    # before running this, it is better to commit or stage the results you want to keep by git
    # make sure you have the correct file name.
    # clear_files("DQN_rb_tn-r1-t20000-bs128-0330_15_38_33.npy")

