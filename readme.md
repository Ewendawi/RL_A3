
Python  3.9.19 was used
## Dependency
- gymnasium
- numpy
- torch
- matplotlib
- scipy
- tensorboardX

## Usage
```shell
python experiment.py -e 0 --device=cpu
```
the output of images will be in the "images" directory

`--disable_multi_process` is used to disable multi-process training.

`--device` is used to specify the device, which can be `cpu`,`cuda` or `cuda:i`. The default value is `cpu`.

`-e` is used to specify the experiment.

## Folder Structure
```
├── common
│   ├── AbstractPGAlgorithm.py (define the abstract class for policy gradient algorithms. you can find what componens are abstracted and the main process of algorithm here)
│   ├── expConfig.py (experiment configuration)
│   ├── helper.py (sample buffer)
│   ├── networks.py (all policy and value networks)
|   ├── plot.py (plotting)
├── AC.py (standard Actor-Critic, like REINFORCE, A2C)
├── experiment.py (run experiments)
├── DDGP.py 
├── PPOClip.py
├── sb3_baseline.py (use sb3 as a baseline)
```
## Features
- REINFORCE ✅
- Actor-Critic ✅
- DDPG ✅
    - noise clip ✅
- PPO-Clip ✅
    - recompute GAE ✅
    - Orthogonal Initialization & Constant Initialization 
    - mini-batch
    - dual-clip ✅
    - g-sde ✅
    - value & gae rescaling, normalization or clip
    - grad clip
    - off-policy ppo
- SAC
- CMS-ES

## Simple Test
| Algorithm | CartPole-v1 | Pendulum-v1 | Ant-v4 |
| --- | --- | --- | --- |
| REINFORCE (200k)| ~350 | ~-1000 | ~-700 |
| Actor-Critic (200k) | ~350 | ~-1000 | ~-800 |
| DDPG (20k)|  | ~-150| 400 |
| PPO-Clip (200k) | ~450 | ~-200 | ~50 |

- CartPole-v1: single discrete action
- Pendulum-v1: single continuous action
- Ant-v4: multiple continuous actions

## Experiments

### Note
- Each experiment should repeat 15 times 
- Results are in dir "results"
- Plot results using "plot_with_file"


## Resources
- https://gymnasium.farama.org/environments/classic_control/
- https://di-engine-docs.readthedocs.io/en/latest/12_policies/index.html
- https://stable-baselines3.readthedocs.io/en/master/index.html
- [The CMA Evolution Strategy: A Tutorial](https://arxiv.org/abs/1604.00772)
- [generalized State Dependent Exploration](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/distributions.py)
- Best Practices:
    - [training-ppo](https://github.com/gzrjzcx/ML-agents/blob/master/docs/Training-PPO.md)
    - [ppo-implementation-details](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) 
    - https://nn.labml.ai/rl/index.html
    - https://portfoly-yoonniverse.vercel.app/post/62f240dc89e6b9dbd5852317/details-and-tricks-to-implement-a-working-ppo
    - [A Closer Look at Deep Policy Gradients](https://arxiv.org/abs/1811.02553)
- Generalized Advantage Estimation:[1](https://arxiv.org/abs/1506.02438), [2](https://towardsdatascience.com/generalized-advantage-estimation-in-reinforcement-learning-bf4a957f7975)
- [Regularization in Reinforcement Learning](https://rl-vs.github.io/rlvs2021/class-material/regularized_mdp/Regularization_RL_RLVS.pdf#page=1.00)
- [How to Initialize Weights in PyTorch](https://wandb.ai/wandb_fc/tips/reports/How-to-Initialize-Weights-in-PyTorch--VmlldzoxNjcwOTg1)