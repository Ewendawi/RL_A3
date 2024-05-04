
import time
from common.expConfig import join_processes, clean_processes, run_experiment
from common.plot import plot_with_file, write_dirs_to_file
from common.AbstractPGAlgorithm import train_model, render_env
from AC import exp_config_for_AC, ActorCritic


def experiment_AC():
    repeat = 1
    standardize_reward = False
    time_steps = 1000000
    batch_size = 0
    n_step = 8

    env_name = "CartPole-v1"
    actor_lr = 1e-3
    critic_lr = 1e-2

    # env_name = "Pendulum-v1"
    # standardize_reward = True
    # reward_uppper_bound = 0.0
    # reward_low_bound = -16.0
    # actor_lr = 1e-4
    # critic_lr = 1e-3

    env_name = "LunarLander-v2"
    batch_size = 32
    actor_lr = 1e-3
    critic_lr = 5e-3 
    n_step = 16

    # env_name = "Ant-v4"
    # standardize_reward = False
    # actor_lr = 1e-4
    # critic_lr = 1e-3 

    exp_name = f"AC-{env_name}-r{repeat}-t{time_steps}"

    exp_config = exp_config_for_AC(exp_name=exp_name, env_name=env_name, repeat=repeat, timesteps=time_steps, device_name=DEVICE_NAME)
    exp_config.actor_config.batch_size = batch_size
    exp_config.actor_config.learning_rate = actor_lr

    exp_config.critic_config.use_base_line = True
    exp_config.critic_config.use_gae = True
    exp_config.critic_config.n_steps = n_step
    exp_config.critic_config.batch_size = batch_size
    exp_config.critic_config.learning_rate = critic_lr
    exp_config.critic_config.standardize_reward = standardize_reward
    if standardize_reward:
        exp_config.critic_config.reward_uppper_bound = reward_uppper_bound
        exp_config.critic_config.reward_low_bound = reward_low_bound
    
    exp_config.tensorboard_dir = f"{exp_name}"

    model = ActorCritic(exp_config.actor_config, exp_config.critic_config)

    result = train_model(env=exp_config.env, model=model, time_steps=exp_config.timesteps, eval_interval=exp_config.eval_interval, eval_env=exp_config.eval_env, eval_episodes=exp_config.eval_episodes, seed=exp_config.seed, tensorboard_dir=exp_config.tensorboard_dir)

    now = time.strftime("%m%d_%H_%M_%S", time.localtime(time.time()))
    model_name = f"AC_{env_name}_r{repeat}_t{time_steps}_{now}"
    model_path = model.save(model_name)
    render_env(model_path)

'''
1. baseline
2. step 
3. learning rate?
4. entropy
'''

def experiment_baseline(multi_process=False, device_name="cpu"):
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
    actor_lr = 1e-3
    critic_lr = 5e-3
    n_step = 16

    exp_name = f"AC-{env_name}-bl-r{repeat}-t{time_steps}"

    exp_config = exp_config_for_AC(exp_name=exp_name, env_name=env_name, repeat=repeat, timesteps=time_steps, device_name=device_name)
    exp_config.critic_config.use_base_line = False
    exp_config.critic_config.use_gae = False
    exp_config.actor_config.learning_rate = actor_lr
    exp_config.critic_config.learning_rate = critic_lr
    exp_config.critic_config.n_steps = n_step
    exp_config.update_dir_name("navie")
    if test_in_tensorboard :
        exp_config.tensorboard_dir = f"{exp_name}-bl-navie"
    run_experiment(exp_config=exp_config, multi_process=run_multi_process)
    dirs.append(exp_config.dir_name)


    # baseline
    exp_config = exp_config_for_AC(exp_name=exp_name, env_name=env_name, repeat=repeat, timesteps=time_steps, device_name=device_name)
    exp_config.critic_config.use_base_line = True
    exp_config.critic_config.use_gae = False
    exp_config.actor_config.learning_rate = actor_lr
    exp_config.critic_config.learning_rate = critic_lr
    exp_config.critic_config.n_steps = n_step
    exp_config.update_dir_name("baseline")
    if test_in_tensorboard :
        exp_config.tensorboard_dir = f"{exp_name}-bl-baseline"
    run_experiment(exp_config=exp_config, multi_process=run_multi_process)
    dirs.append(exp_config.dir_name)

    # gae
    exp_config = exp_config_for_AC(exp_name=exp_name, env_name=env_name, repeat=repeat, timesteps=time_steps, device_name=device_name)
    exp_config.critic_config.use_base_line = False
    exp_config.critic_config.use_gae = True
    exp_config.actor_config.learning_rate = actor_lr
    exp_config.critic_config.learning_rate = critic_lr
    exp_config.critic_config.n_steps = n_step
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
    


def experiment_steps(multi_process=False, device_name="cpu"):
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

    actor_lr = 1e-3
    critic_lr = 5e-3 
    steps = [8, 32, 128]

    exp_name = f"AC-{env_name}-step-r{repeat}-t{time_steps}"
    for step in steps:
        exp_config = exp_config_for_AC(exp_name=exp_name, env_name=env_name, repeat=repeat, timesteps=time_steps, device_name=device_name)
        exp_config.critic_config.use_base_line = True
        exp_config.critic_config.use_gae = False
        exp_config.actor_config.learning_rate = actor_lr
        exp_config.critic_config.learning_rate = critic_lr
        exp_config.critic_config.n_steps = step

        exp_config.update_dir_name(str(step))

        if test_in_tensorboard:
            exp_config.tensorboard_dir = f"{exp_name}-step{step}"

        run_experiment(exp_config=exp_config, multi_process=run_multi_process)
        dirs.append(exp_config.dir_name)

    if run_multi_process:
        join_processes()
        clean_processes()
    
    if test_in_tensorboard:
        return

    exp_labels = [str(step) for step in steps]
    file = write_dirs_to_file(dirs=dirs, labels=exp_labels, file_name=exp_name)

    names = [f"{exp_name}_train", f"{exp_name}_eval"]
    plot_with_file(file, names=names)

# experiment_steps()

def experiment_entropy(multi_process=False, device_name="cpu"):
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
    actor_lr = 1e-3
    critic_lr = 5e-3 
    n_step = 16

    entropies = [0, 0.1, 0.3]

    exp_name = f"AC-{env_name}-en-r{repeat}-t{time_steps}"
    for entropy in entropies:
        exp_config = exp_config_for_AC(exp_name=exp_name, env_name=env_name, repeat=repeat, timesteps=time_steps, device_name=device_name)
        exp_config.critic_config.use_base_line = True
        exp_config.critic_config.use_gae = False
        exp_config.actor_config.learning_rate = actor_lr
        exp_config.critic_config.learning_rate = critic_lr
        exp_config.critic_config.n_steps = n_step
        exp_config.actor_config.entropy_weight = entropy

        exp_config.update_dir_name(str(entropy))

        if test_in_tensorboard :
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
    
#added:
def experiment_AC_entropy_weights():
    #YES bootstrapping
    #YES baseline substraction
    
    dirs = []
    repeat = 5
    run_multi_process = USE_MULTI_PROCESS and False
    time_steps = 1000000  
    n_step=16
    env_name = "LunarLander-v2"
    entropy_weights = [0.0001, 0.01, 0.1]
    exp_name = f"AC-{env_name}-r{repeat}-t{time_steps}"

    for entropy_weight in entropy_weights:
        exp_config = exp_config_for_AC(exp_name=exp_name, env_name=env_name, repeat=repeat, timesteps=time_steps, device_name=DEVICE_NAME)
        exp_config.actor_config.entropy_weight = entropy_weight
        exp_config.critic_config.use_base_line = True
        exp_config.critic_config.n_steps = n_step
        exp_config.tensorboard_dir = f"{exp_name}_entropy_weight_{entropy_weight}"
        exp_config.update_dir_name(suffix=f"entropy_weight_{entropy_weight}")
        model = ActorCritic(exp_config.actor_config, exp_config.critic_config)
        run_experiment(model, exp_config=exp_config, multi_process=False)
        dirs.append(exp_config.dir_name)

    if run_multi_process:
        join_processes()
        clean_processes()

    exp_labels = ["entropy_weight_" + str(weight) for weight in entropy_weights]
    file = write_dirs_to_file(dirs=dirs, labels=exp_labels, file_name=exp_name)

    names = [f"{exp_name}_train", f"{exp_name}_eval"]
    plot_with_file(file, names=names)

def experiment_AC_critic_lr():
    #YES bootstrapping
    #YES baseline substraction
    
    dirs = []
    repeat = 5
    run_multi_process = USE_MULTI_PROCESS and False
    time_steps = 1000000  
    n_step=16
    env_name = "LunarLander-v2"
    learning_rates = [0.001, 0.01, 0.1]
    exp_name = f"AC-{env_name}-r{repeat}-t{time_steps}"
    entropy_weight = 0.001

    for learning_rate in learning_rates:
        exp_config = exp_config_for_AC(exp_name=exp_name, env_name=env_name, repeat=repeat, timesteps=time_steps, device_name=DEVICE_NAME)
        exp_config.critic_config.learning_rate = learning_rate
        exp_config.actor_config.entropy_weight = entropy_weight
        exp_config.critic_config.use_base_line = True
        exp_config.critic_config.n_steps = n_step
        exp_config.tensorboard_dir = f"{exp_name}_learning_rate_c{learning_rate}"
        exp_config.update_dir_name(suffix=f"learning_rate_c{learning_rate}")
        model = ActorCritic(exp_config.actor_config, exp_config.critic_config)
        run_experiment(model, exp_config=exp_config, multi_process=False)
        dirs.append(exp_config.dir_name)

    if run_multi_process:
        join_processes()
        clean_processes()

    exp_labels = ["learning_rate_c" + str(learning_rate) for learning_rate in learning_rates]
    file = write_dirs_to_file(dirs=dirs, labels=exp_labels, file_name=exp_name)

    names = [f"{exp_name}_train", f"{exp_name}_eval"]
    plot_with_file(file, names=names)

def experiment_AC_actor_lr():
    #YES bootstrapping
    #YES baseline substraction
    
    dirs = []
    repeat = 5
    run_multi_process = USE_MULTI_PROCESS and False
    time_steps = 1000000  
    n_step=16
    env_name = "LunarLander-v2"
    learning_rates = [0.001, 0.01, 0.1]
    exp_name = f"AC-{env_name}-r{repeat}-t{time_steps}"
    entropy_weight = 0.001

    for learning_rate in learning_rates:
        exp_config = exp_config_for_AC(exp_name=exp_name, env_name=env_name, repeat=repeat, timesteps=time_steps, device_name=DEVICE_NAME)
        exp_config.actor_config.learning_rate = learning_rate
        exp_config.actor_config.entropy_weight = entropy_weight
        exp_config.critic_config.use_base_line = True
        exp_config.critic_config.n_steps = n_step
        exp_config.tensorboard_dir = f"{exp_name}_learning_rate_a{learning_rate}"
        exp_config.update_dir_name(suffix=f"learning_rate_a{learning_rate}")
        model = ActorCritic(exp_config.actor_config, exp_config.critic_config)
        run_experiment(model, exp_config=exp_config, multi_process=False)
        dirs.append(exp_config.dir_name)

    if run_multi_process:
        join_processes()
        clean_processes()

    exp_labels = ["learning_rate_a" + str(learning_rate) for learning_rate in learning_rates]
    file = write_dirs_to_file(dirs=dirs, labels=exp_labels, file_name=exp_name)

    names = [f"{exp_name}_train", f"{exp_name}_eval"]
    plot_with_file(file, names=names)

def experiment_AC_nstep():
    #YES bootstrapping
    #YES baseline substraction
    
    dirs = []
    repeat = 5
    run_multi_process = USE_MULTI_PROCESS and False
    time_steps = 1000000  
    n_step=16
    env_name = "LunarLander-v2"
    nsteps = [5, 16, 50]
    exp_name = f"AC-{env_name}-r{repeat}-t{time_steps}"
    entropy_weight = 0.001

    for nstep in nsteps:
        exp_config = exp_config_for_AC(exp_name=exp_name, env_name=env_name, repeat=repeat, timesteps=time_steps, device_name=DEVICE_NAME)
        exp_config.actor_config.nstep = nstep
        exp_config.actor_config.entropy_weight = entropy_weight
        exp_config.critic_config.use_base_line = True
        exp_config.critic_config.n_steps = n_step
        exp_config.tensorboard_dir = f"{exp_name}_nstep_{nstep}"
        exp_config.update_dir_name(suffix=f"nstep_{nstep}")
        model = ActorCritic(exp_config.actor_config, exp_config.critic_config)
        run_experiment(model, exp_config=exp_config, multi_process=False)
        dirs.append(exp_config.dir_name)

    if run_multi_process:
        join_processes()
        clean_processes()

    exp_labels = ["nstep" + str(nstep) for nstep in nsteps]
    file = write_dirs_to_file(dirs=dirs, labels=exp_labels, file_name=exp_name)

    names = [f"{exp_name}_train", f"{exp_name}_eval"]
    plot_with_file(file, names=names)


def experiment_AC_base_boot():
    #experiment with/without bootstrapping and baseline
    
    dirs = []
    repeat = 5
    run_multi_process = USE_MULTI_PROCESS and False
    time_steps = 1000000  
    n_step=16
    env_name = "LunarLander-v2"
    exp_name = f"AC-{env_name}-r{repeat}-t{time_steps}"
    entropy_weight = 0.001

    flugs = [[True,True],[False,False],[True,False],[False,True]]

    for i in range(4):
        exp_config = exp_config_for_AC(exp_name=exp_name, env_name=env_name, repeat=repeat, timesteps=time_steps, device_name=DEVICE_NAME)
        exp_config.actor_config.entropy_weight = entropy_weight

        exp_config.critic_config.use_base_line = flugs[i][0]
        exp_config.critic_config.n_steps = int(flugs[i][1]*n_step)

        exp_config.tensorboard_dir = f"{exp_name}base_boot_{flugs[i]}"
        exp_config.update_dir_name(suffix=f"base_boot_{flugs[i]}")
        model = ActorCritic(exp_config.actor_config, exp_config.critic_config)
        run_experiment(model, exp_config=exp_config, multi_process=False)
        dirs.append(exp_config.dir_name)

    if run_multi_process:
        join_processes()
        clean_processes()

    exp_labels = ["base_boot" + str(flug) for flug in flugs]
    file = write_dirs_to_file(dirs=dirs, labels=exp_labels, file_name=exp_name)

    names = [f"{exp_name}_train", f"{exp_name}_eval"]
    plot_with_file(file, names=names)

