import time
from stable_baselines3 import A2C, PPO, DDPG
from stable_baselines3.common.env_util import make_vec_env
import numpy as np

def A2C_baseline():
    # env_name = "Pendulum-v1"
    # env_name = "CartPole-v1"
    env_name = "Ant-v4"

    now = time.strftime("%m%d_%H_%M_%S", time.localtime(time.time()))
    file_path = f"./models/sb3_a2c_{env_name}_{now}"
    
    vec_env = make_vec_env(env_name, n_envs=4)

    model = A2C("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=200000)
    model.save(file_path)

    # del model # remove to demonstrate saving and loading

    # file_path = "a2c_cartpole.zip"
    model = A2C.load(file_path)

    obs = vec_env.reset()
    r = 0
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")

        r += rewards[0]
        if dones[0] :
            obs = vec_env.reset()
            print (r)
            r = 0

def DDPG_baseline():
    env_name = "Pendulum-v1"
    env_name = "Ant-v4"

    now = time.strftime("%m%d_%H_%M_%S", time.localtime(time.time()))
    file_path = f"./models/sb3_ddpg_{env_name}_{now}"
    
    vec_env = make_vec_env(env_name, n_envs=4)

    model = DDPG("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=20000)
    model.save(file_path)

    # del model # remove to demonstrate saving and loading

    # file_path = "a2c_cartpole.zip"
    model = DDPG.load(file_path)

    obs = vec_env.reset()
    r = 0
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")

        r += rewards[0]
        if dones[0] :
            obs = vec_env.reset()
            print (r)
            r = 0

def PPO_baseline():
    env_name = "Pendulum-v1"
    # env_name = "CartPole-v1"
    env_name = "Ant-v4"
    env_name = "MountainCar-v0"
    # env_name = "MountainCarContinuous-v0"

    now = time.strftime("%m%d_%H_%M_%S", time.localtime(time.time()))
    file_path = f"./models/sb3_ppo_{env_name}_{now}"
    
    vec_env = make_vec_env(env_name, n_envs=1)

    model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="./logs/")
    model.learn(total_timesteps=1000000)
    model.save(file_path)

    # del model # remove to demonstrate saving and loading

    # file_path = "a2c_cartpole.zip"
    model = PPO.load(file_path)

    obs = vec_env.reset()
    r = 0
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")

        r += rewards[0]
        if dones[0] :
            obs = vec_env.reset()
            print (r)
            r = 0

# A2C_baseline()
PPO_baseline()
# DDPG_baseline()