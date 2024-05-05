
import sys
import getopt

from common.plot import plot_with_file, clear_files, plot_file_with_keys
from common.AbstractPGAlgorithm import render_env
import exp_reinforce
import exp_AC
import exp_ppo

USE_MULTI_PROCESS = True
DEVICE_NAME = "cpu"
# if torch.backends.mps.is_available():
#     DEVICE_NAME = "mps"

if __name__ == "__main__":
    argv = sys.argv[1:]
    experiment = 0
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

    if experiment == 0:
        pass
    # reinforce
    # 10: test reinforce
    # 11: learning rate
    # 12: batch size
    # 13: entropy
    # 14: baseline
    elif experiment == 10:
        exp_reinforce.experiment_reinforce()
    elif experiment == 11:
        exp_reinforce.experiment_learing_rate(multi_process=USE_MULTI_PROCESS, device_name=DEVICE_NAME)
    elif experiment == 12:
        exp_reinforce.experiment_batch_size(multi_process=USE_MULTI_PROCESS, device_name=DEVICE_NAME)
    elif experiment == 13:
        exp_reinforce.experiment_entropy(multi_process=USE_MULTI_PROCESS, device_name=DEVICE_NAME)
    elif experiment == 14:
        exp_reinforce.experiment_baseline(multi_process=USE_MULTI_PROCESS, device_name=DEVICE_NAME)
    # Actor Critic
    # 20: test AC
    # 21: baseline
    # 22: learning rate
    # 23: entropy
    # 24: steps
    elif experiment == 20:
        exp_AC.experiment_AC()
    elif experiment == 21:
        exp_AC.experiment_baseline(multi_process=USE_MULTI_PROCESS, device_name=DEVICE_NAME)
    elif experiment == 22:
        pass
    elif experiment == 23:
        exp_AC.experiment_entropy(multi_process=USE_MULTI_PROCESS, device_name=DEVICE_NAME)
    elif experiment == 24:
        exp_AC.experiment_steps(multi_process=USE_MULTI_PROCESS, device_name=DEVICE_NAME)
    # PPO
    # 30: test PPO
    # 31: PPO lr
    # 32: PPO kl_div
    # 33: PPO repeat
    elif experiment == 30:
        exp_ppo.experiment_PPO()
    elif experiment == 31:
        exp_ppo.experiment_ppo_lr(multi_process=USE_MULTI_PROCESS, device_name=DEVICE_NAME)
    elif experiment == 32:
        exp_ppo.experiment_ppo_kl(multi_process=USE_MULTI_PROCESS, device_name=DEVICE_NAME)
    elif experiment == 33:
        exp_ppo.experiment_ppo_repeat(multi_process=USE_MULTI_PROCESS, device_name=DEVICE_NAME)
    # utils
    # 100: render
    # 101: plot
    # 102: clear
    elif experiment == 100:
        path = "./models/AC_Pendulum-v1_True_r1_t200000_0412_23_45_31"
        render_env(path)
    elif experiment == 101:
        path = "experiments/PPO-LunarLander-v2-lr-r5-t1000000-0504_23_28_27.npy"
        names = ["train", "eval"]
        names = []
        plot_with_file(path, names=names)
    elif experiment == 102:
        # clear the results you don't need.
        # before running this, it is better to commit or stage the results you want to keep by git
        # make sure you have the correct file name.
        file_path = "experiments/reinforce-LunarLander-v2-lr-r10-t100000-0504_11_39_06.npy"
        clear_files(file_path)
    elif experiment == 103:
        keys = {
            "eval": ["eval", "eval"],
            "critic_loss": ["critic_loss", "critic_loss"],
            "approx_kl_div": ["kl_div", "approx_kl_div"],
        }
        path = "experiments/PPO-LunarLander-v2-repeat-r5-t1000000-0505_00_35_22.npy"
        plot_file_with_keys(path, keys)  

    
    elif experiment == 29:
        print('Running experiment actor-critic entropy_weights')
        exp_AC.experiment_AC_entropy_weights(multi_process=USE_MULTI_PROCESS, device_name=DEVICE_NAME)

    elif experiment == 1926:
        print('Running experiment actor-critic critic_lr')
        exp_AC.experiment_AC_critic_lr(multi_process=USE_MULTI_PROCESS, device_name=DEVICE_NAME)

    elif experiment == 1979:
        print('Running experiment actor-critic actor_lr')
        exp_AC.experiment_AC_actor_lr(multi_process=USE_MULTI_PROCESS, device_name=DEVICE_NAME)

    elif experiment == 2000:
        print('Running experiment actor-critic nstep')
        exp_AC.experiment_AC_nstep(multi_process=USE_MULTI_PROCESS, device_name=DEVICE_NAME)

    elif experiment == 2024:
        print('Running experiment actor-critic base_boot')
        exp_AC.experiment_AC_base_boot(multi_process=USE_MULTI_PROCESS, device_name=DEVICE_NAME )

    elif experiment == 42:
        print('Running experiment actor-critic nstep-no baseline')
        exp_AC.experiment_AC_nstep_nobase(multi_process=USE_MULTI_PROCESS, device_name=DEVICE_NAME )


