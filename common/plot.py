import os
import pathlib
import shutil
import re
import numpy as np
import time
from typing import List
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def is_directory_empty(dir):
    with os.scandir(dir) as it:
        return not any(it)


def get_empty_dir():
    data_dir = pathlib.Path("./results")
    dirlist = list(data_dir.glob("*"))
    for dir in dirlist:
        if dir.is_dir and is_directory_empty(dir):
            # os.rmdir(dir)
            print(dir)


def clear_files(file_name):
    file_path = f"./{file_name}"
    data = np.load(file_path, allow_pickle=True)
    dirs = data.item().get("dirs")
    for dir in dirs:
        data_dir = pathlib.Path("./results/" + dir)
        shutil.rmtree(data_dir, ignore_errors=True)
    os.remove(file_path)


def clear_files_with_re():
    data_dir = pathlib.Path(".")
    file_list = list(data_dir.glob("*rb_size*.npy"))
    for file_path in file_list:
        filename_with_ext = os.path.basename(file_path)
        clear_files(filename_with_ext)


def get_mean_value_with_filter(
    filters, keys=["eval", "train", "time_steps", "loss"], smooth_window=0
):
    data_dir = pathlib.Path("./results")
    dirlist = list(data_dir.glob("*"))

    results = {}
    for f in filters:
        # find all the directories that match the filter
        def filter_func(x):
            return re.search(f, str(x))

        target_dirlist = list(filter(filter_func, dirlist))
        if not target_dirlist or len(target_dirlist) > 1:
            continue
        target_dir = target_dirlist[0]

        file_data = (
            {}
        )  # { "return":{0: np.array, 1: np.array, ...}, "time_steps": {0: np.array, 1: np.array, ...}
        for key in keys:
            dataMap = {}
            filelist = list(target_dir.glob(f"*{key}*.npy"))
            key_data = {}
            for file in filelist:
                filename_with_ext = os.path.basename(file)
                repetation = filename_with_ext.split("-")[0]
                key_data[repetation] = np.load(file)
            file_data[key] = key_data
        results[f] = file_data

    res = {}
    for key, file in results.items():
        key_data = {}
        for key_t, value_t in file.items():
            data = np.array(list(value_t.values()))
            mean_data = np.mean(data, axis=0)
            max_data = np.max(data, axis=0)
            min_data = np.min(data, axis=0)
            std_data = np.std(data, axis=0)
            if smooth_window > 0 and mean_data.size > 2 * smooth_window + 1:
                max_data = savgol_filter(max_data, smooth_window, 2)
                mean_data = savgol_filter(mean_data, smooth_window, 2)
                min_data = savgol_filter(min_data, smooth_window, 2)
                std_data = savgol_filter(std_data, smooth_window, 2)
            key_data[key_t] = (mean_data, max_data, min_data, std_data)
        res[key] = key_data

    return res


def plot_with_data(res_map, exp_labels=None, x_y_labels=None, save_name=None):
    fig, ax = plt.subplots()

    labels = exp_labels if exp_labels else res_map.keys()
    label_index = 0

    for exp, exp_data in res_map.items():
        label_name = labels[label_index]
        x = exp_data["time_steps"][0]
        for key, data in exp_data.items():
            if key == "time_steps":
                continue
            mean_data, max_data, min_data, std_data = data
            ax.plot(x, mean_data, label=label_name)
            ax.fill_between(x, min_data, max_data, alpha=0.2)
        label_index += 1

    # ax.set_ylim(0, ylim)
    if x_y_labels:
        plt.xlabel(x_y_labels[0])
        plt.ylabel(x_y_labels[1])
    plt.grid(True)
    plt.legend(
        # loc='upper left',
        prop={"size": 8},
        # bbox_to_anchor=(0.5, 1.15),
        # ncol=3
    )
    if save_name:
        plots_dir = "./images"
        if not os.path.isdir(plots_dir):
            os.makedirs(plots_dir)
        now = time.strftime("%m%d_%H_%M_%S", time.localtime(time.time()))
        time_name = save_name + "-" + now
        plt.savefig(f"{plots_dir}/{time_name}", dpi=300)
    else:
        plt.show(block=True)


def plot_with_dirs(
    filters,
    keys=None,
    smooth_window=0,
    exp_labels=None,
    x_y_labels=None,
    save_name=None,
):
    res_map = get_mean_value_with_filter(filters, keys, smooth_window)
    plot_with_data(res_map, exp_labels, x_y_labels, save_name=save_name)


def plot_rb_tn(dirs: List[str] = [], exp_labels: List[str] = [], names=None):
    filters = dirs

    name = names[0] if names else None

    keys = ["train", "time_steps"]
    plot_with_dirs(
        filters=filters,
        keys=keys,
        smooth_window=9,
        exp_labels=exp_labels,
        x_y_labels=["time_steps", "return"],
        save_name=name,
    )

    name = names[1] if names else None
    keys = ["eval", "time_steps"]
    plot_with_dirs(
        filters=filters,
        keys=keys,
        smooth_window=9,
        exp_labels=exp_labels,
        x_y_labels=["time_steps", "return"],
        save_name=name,
    )

    # keys = ['time_steps', 'loss']
    # name = names[2] if names else None
    # plot_with_dirs(filters=filters, keys=keys, smooth_window=9, exp_labels=exp_labels, x_y_labels=["time_steps", "loss"])


def plot_with_file(file_path, names=None):
    data = np.load(file_path, allow_pickle=True)
    dirs = data.item().get("dirs")
    labels = data.item().get("labels")
    plot_rb_tn(dirs=dirs, exp_labels=labels, names=names)


def write_dirs_to_file(dirs: List[str] = [], labels=[], file_name=""):
    experiments_dir = "./experiments"
    if not os.path.isdir(experiments_dir):
        os.makedirs(experiments_dir)

    now = time.strftime("%m%d_%H_%M_%S", time.localtime(time.time()))
    experiment_name = file_name + "-" + now

    file_path = f"{experiments_dir}/{experiment_name}.npy"
    data = {"dirs": dirs, "labels": labels}
    np.save(file_path, data)
    return file_path
