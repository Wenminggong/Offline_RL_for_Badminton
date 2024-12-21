# -*- coding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2024/07/21 19:58:45
@Author  :   Mingjiang Liu 
@Version :   1.0
@Desc    :   useful functions for win rate prediction
'''

import numpy as np
import matplotlib.pyplot as plt
import os


def convert_rally_to_whole(trajectories):
    # design for MLP-based win-no_win prediction
    # convert original dataset [rally, rally, ...] to {key: ndarray}
    dataset = {}
    for rally in trajectories:
        rally["reward"][:-1] = 0.0
        if rally["hit_xy"][0].sum() == 0:
            # serve action, delete
            for key in rally.keys():
                if isinstance(rally[key], np.ndarray):
                    rally[key] = rally[key][1:]

        if rally["reward"][-1] == 0:
            # last action is no win, delete
            for key in rally.keys():
                if isinstance(rally[key], np.ndarray):
                    rally[key] = rally[key][:-1]

        for key in rally.keys():
            if key in dataset.keys():
                dataset[key].append(rally[key])
            else:
                dataset[key] = [rally[key]]

    for key in rally.keys():
        if isinstance(rally[key], np.ndarray):
            dataset[key] = np.concatenate(dataset[key], axis=0)

    # value: [batch_size]-ndarray, or [batch_size, n]-ndarray, or List[]
    return dataset


def show_prob_distribution(data, path, name, show=True):

    # 计算平均值和标准差
    keys = list(data.keys())
    means = [np.mean(data[key]) for key in keys]
    std_devs = [np.std(data[key]) for key in keys]

    # 绘制箱线图
    fig, ax = plt.subplots()

    # 绘制箱线图
    ax.boxplot([data[key] for key in keys], labels=keys)

    # 添加平均值和标准差
    for i, key in enumerate(keys):
        ax.errorbar(i + 1, means[i], yerr=std_devs[i], fmt='o', color='red', capsize=5)

    # 设置标题和标签
    ax.set_title('{} - Probability boxplot.'.format(name))
    ax.set_xlabel('Ball round')
    ax.set_ylabel('Probability')

    # 显示图形
    if show:
        plt.show()
    plt.savefig(os.path.join(path, "{}_proba_dist.png".format(name)))