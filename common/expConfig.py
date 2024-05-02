
import os, time
import numpy as np
import torch.multiprocessing as mp
from common.AbstractPGAlgorithm import train_model, ActorConfig, CriticConfig, AbstractPGAlgorithm

PROCESS_QUEUE:list[mp.Process] = []


class ExpConfig:
    def __init__(self, exp_name, repeat, env, timesteps, eval_interval, eval_env, eval_episodes=20):
        self.env = env
        self.timesteps = timesteps

        self.exp_name = exp_name
        self.dir_name: str 

        self.repeat = repeat

        self.eval_interval = eval_interval
        self.eval_env = eval_env
        self.eval_episodes = eval_episodes

        self.seed = 123

        self.actor_config: ActorConfig 
        self.critic_config: CriticConfig 

        self.tensorboard_dir = None

    def update_dir_name(self, suffix:str=""):
        self.dir_name = f"{self.exp_name}-{suffix}-{time.time()}"

def join_processes() -> None:
    for p in PROCESS_QUEUE:
        p.join()

def clean_processes():
    for p in PROCESS_QUEUE:
        p.terminate()
    PROCESS_QUEUE.clear()


# Note: this is for macos. If you have fork error, export this in terminal
"""
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
"""
# Default start method is 'spawn' on macos. but it have pickling error.
# It can be solved by removing function block.

# For fork method
# If pytorch's autograd engine is initialized before fork, it will not work.
# Because autograd engine relies on threads pool, which makes it vulnerable to fork(memory space problem).
# That means, you should not use multiprocessing in release mode or run multi-exepriments at the same time.
# https://github.com/pytorch/pytorch/wiki/Autograd-and-Fork
# https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
# https://github.com/pytorch/pytorch/blob/main/torch/csrc/autograd/engine.cpp

if mp.get_context().get_start_method() == 'fork':
    mp.set_start_method("spawn")

def task(model, exp_config) -> None:
    dir_name = exp_config.dir_name
    repeat = exp_config.repeat
    # make dir results
    dir_path = f"./results/{dir_name}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    for i in range(repeat):
        print(f"Running {i+1} of {repeat}")
        start = time.time()
        results = train_model(env=exp_config.env, model=model, 
                              time_steps=exp_config.timesteps, 
                              eval_interval=exp_config.eval_interval, eval_env=exp_config.eval_env, eval_episodes=exp_config.eval_episodes, 
                              seed=exp_config.seed, tensorboard_dir=exp_config.tensorboard_dir)
        print(f"Time taken: {time.time() - start:.2f}s")

        for key, value in results.items():
            file_path = f"{dir_path}/{i}-{key}.npy"  
            np.save(file_path, value)
            print(f"saved {file_path}")

def run_experiment(model, exp_config, multi_process=False):
    if multi_process:
        p = mp.Process(target=task, args=(model,exp_config,))
        PROCESS_QUEUE.append(p)
        p.start()
    else:
        task(model, exp_config)
