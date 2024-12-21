# -*- coding: utf-8 -*-
'''
@File    :   visualization.py
@Time    :   2024/04/17 10:36:00
@Author  :   Mingjiang Liu 
@Version :   1.0
@Desc    :   for visualization.
'''


import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image
import argparse
from preprocess_badminton_data import ACTIONS


def visualization(data_path, save_path, show_pred_action:True):
    save_path = os.path.join(save_path, os.path.split(data_path)[1].split(".")[0])
    data = pd.read_csv(data_path, converters={"ball": eval,"top":eval,"bottom":eval,"court":eval,"net":eval})

    court=data.loc[0,'court']
    net=data.loc[0,'net']

    image_list=[]
    for index,row in data.iterrows():
        if not row["type"] in ACTIONS.keys():
            continue

        plt.figure(figsize=(16, 12)) 
        plt.ylim(0, 1080)
        plt.xlim(0, 1920)
        plt.gca().invert_yaxis()

        # 给定的点
        players_joints=row['top']

        # 提取 x 坐标和 y 坐标
        x = [joint[0] for joint in players_joints]
        y = [joint[1] for joint in players_joints]

        # 创建散点图
        plt.scatter(x, y, c="b")

        edges = [(0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (11, 12),
                (5, 7), (7, 9), (5, 11), (11, 13), (13, 15), (6, 12),
                (12, 14), (14, 16), (5, 6)]

        # 循环添加标号
        for i, joint in enumerate(players_joints):
            plt.annotate(str(i), (x[i], y[i]), textcoords="offset points", xytext=(5,5), ha='center')

        # 绘制连接线
        for edge in edges:
            plt.plot([x[edge[0]], x[edge[1]]], [y[edge[0]], y[edge[1]]], 'b-')

        players_joints=row['bottom']

        # 提取 x 坐标和 y 坐标
        x = [joint[0] for joint in players_joints]
        y = [joint[1] for joint in players_joints]

        # 创建散点图
        plt.scatter(x, y, c="r")

        # 循环添加标号
        for i, joint in enumerate(players_joints):
            plt.annotate(str(i), (x[i], y[i]), textcoords="offset points", xytext=(0,10), ha='center')

        # 绘制连接线
        for edge in edges:
            plt.plot([x[edge[0]], x[edge[1]]], [y[edge[0]], y[edge[1]]], 'r-')


        # 场
        # 提取 x 坐标和 y 坐标
        x = [joint[0] for joint in court]
        y = [joint[1] for joint in court]
        # 创建散点图
        plt.scatter(x, y, c="y")
        edges = [(0, 1), (2, 3), (4, 5),(0,4),(1,5)]
        # 循环添加标号
        for i, joint in enumerate(court):
            plt.annotate(str(i), (x[i], y[i]), textcoords="offset points", xytext=(0,10), ha='center')
        # 绘制连接线
        for edge in edges:
            plt.plot([x[edge[0]], x[edge[1]]], [y[edge[0]], y[edge[1]]], 'y-')


        # 网
        # 提取 x 坐标和 y 坐标
        x = [joint[0] for joint in net]
        y = [joint[1] for joint in net]
        # 创建散点图
        plt.scatter(x, y,c="y")
        edges = [(0, 1), (1, 2), (2, 3),(0,3)]
        # 循环添加标号
        for i, joint in enumerate(net):
            plt.annotate(str(i), (x[i], y[i]), textcoords="offset points", xytext=(0,10), ha='center')
        # 绘制连接线
        for edge in edges:
            plt.plot([x[edge[0]], x[edge[1]]], [y[edge[0]], y[edge[1]]], 'y-')

        # 球
        ball=row['ball']
        plt.scatter(ball[0], ball[1], c="purple")
        plt.annotate("ball", (ball[0], ball[1]), textcoords="offset points", xytext=(0,10), ha='center')

        # 设置图形标题和轴标签
        plt.title(os.path.split(data_path)[1].split(".")[0])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.text(100, 100, "action: " + row["type"], fontdict={"size":20, "color":"black"})
        if show_pred_action:
            if row["pred_action"] != row["type"]:
                plt.text(100, 200, "** pred action: " + row["pred_action"] + " **", fontdict={"size":20, "color":"red"})
            else:
                plt.text(100, 200, "pred action: " + row["pred_action"], fontdict={"size":20, "color":"red"})

        # 使用PIL库加载图像文件，并将其添加到图像列表中
        print(save_path)
        plt.savefig(save_path+".png")
        image_list.append(Image.open(save_path+".png").copy())

        plt.clf()
        plt.close()

    os.remove(save_path+".png")
    # 保存为GIF文件
    image_list[0].save(save_path+".gif", save_all=True, append_images=image_list[1:], duration=5000, loop=0)


def main(args):
    dataset_path = os.path.join(os.path.dirname(__file__), args.dataset_path)
    save_path = os.path.join(os.path.dirname(__file__), args.save_path)
    assert os.path.exists(dataset_path), "not found dataset!"

    dataset_list = os.listdir(dataset_path)
    i = 1
    for dataset in dataset_list:
        # if not "Akane_YAMAGUCHI_AN_Se_Young_DAIHATSU_YONEX_Japan_Open_2022_Finals" in dataset:
        #     continue
        real_save_path = os.path.join(save_path, dataset)
        if not os.path.exists(real_save_path):
            os.mkdir(real_save_path)
        real_path = os.path.join(dataset_path, dataset)
        rally_list = os.listdir(real_path)
        j = 1
        for rally in rally_list:
            if not ".csv" in rally:
                continue
            print("="*20)
            print("game-rally: {} - {}".format(i, j))
            rally_path = os.path.join(real_path, rally)
            visualization(rally_path, real_save_path, False)
            j += 1
        i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="visualization")
    parser.add_argument("--save_path", type=str, default="visualization")
    args = parser.parse_args()
    main(args)