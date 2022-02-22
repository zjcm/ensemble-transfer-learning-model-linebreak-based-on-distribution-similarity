import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpathes
from main import data_scenes,N

generate_picture_0 = 0
generate_picture_1 = 0
generate_picture_2 = 1
generate_picture_3 = 0
generate_picture_4 = 0

if generate_picture_0: # 预测效果实验

    fig = plt.figure(dpi=300)
    ax1 = fig.add_subplot(111)
    size = 3
    plt.xlabel('data-scene-index')
    plt.ylabel('R-square')
    plt.ylim(-1, 1)
    plt.xlim(0, 17)
    plt.xticks(list(range(17)))
    method_num = 4
    color_names = ['black', 'red', 'green', 'purple']

    for i in data_scenes:

        with open('dataUtest/data' + str(i) + '.txt', 'r') as file:
            r2s = eval(file.read())        


        X = [_ / N / 2 + 0.75 + i - 1 for _ in range(N)]
        Y = r2s
        YT = [[row[i] for row in Y] for i in range(len(Y[0]))]
        for m in range(method_num):
            ax1.scatter(X, YT[m], c=color_names[m], alpha=0.75, s=size)

        # plt.grid(axis="x",c='r',linestyle='-.')
        if i == 1:
            plt.legend(['baseline', 'proposed', 'global-randomforest', 'task-bagging'])
        if i % 2 == 1:
            rect = mpathes.Rectangle((0.5 + i - 1, -15), 1, 30, color='grey', alpha=0.3)
            ax1.add_patch(rect)
        elif i % 2 == 0:
            rect = mpathes.Rectangle((0.5 + i - 1, -15), 1, 30, color='royalblue', alpha=0.3)
            ax1.add_patch(rect)
        for m in range(method_num):
            plt.plot([i - 0.5, i + 0.5], [np.mean(YT[m]), np.mean(YT[m])], c=color_names[m], linestyle='--',
                     linewidth=size / 3.5)
    plt.savefig("trailUtest_fig.png")
    plt.show()

if generate_picture_1:# 噪声实验

    fig = plt.figure(dpi=300)
    ax1 = fig.add_subplot(111)
    size = 3
    plt.xlabel('number of U')
    plt.ylabel('R-square')
    plt.ylim(-1, 1)
    plt.xlim(-0.5, 16)
    plt.xticks(list(range(17)))
    method_num = 3
    color_names = ['black', 'red', 'purple']

    for i in range(16):

        with open('dataUtest/data' + str(i) + '.txt', 'r') as file:
            r2s = eval(file.read())        


        X = [_ / N / 2 + 0.75 + i - 1 for _ in range(N)]
        Y = r2s
        YT = [[row[i] for row in Y] for i in range(len(Y[0]))]
        for m in [0,1,2]:
            ax1.scatter(X, YT[m], c=color_names[m], alpha=0.75, s=size)

        # plt.grid(axis="x",c='r',linestyle='-.')
        if i == 0:
            plt.legend(['baseline', 'proposed', 'task-bagging'])
        if i % 2 == 1:
            rect = mpathes.Rectangle((0.5 + i - 1, -15), 1, 30, color='grey', alpha=0.3)
            ax1.add_patch(rect)
        elif i % 2 == 0:
            rect = mpathes.Rectangle((0.5 + i - 1, -15), 1, 30, color='royalblue', alpha=0.3)
            ax1.add_patch(rect)
        for m in [0,1,2]:
            plt.plot([i - 0.5, i + 0.5], [np.median(YT[m]), np.median(YT[m])], c=color_names[m], linestyle='--',
                     linewidth=size / 3.5)
    plt.savefig("trailUtest_fig.png")
    plt.show()

if generate_picture_2:# C值-集成策略 实验

    fig = plt.figure(dpi=300)
    ax1 = fig.add_subplot(111)
    size = 3
    plt.xlabel('value of c')
    plt.ylabel('R-square')
    plt.ylim(-1, 1)
    plt.xlim(-0.4, 1)
    plt.xticks([K/20 for K in range(-6,19,2)])
    method_num = 1
    color_names = ['black', 'red', 'purple']

    flag = 0
    for i in [K/20 for K in range(-6,19,2)]:

        with open('dataCtest/data_r2s_' + str(i) + '.txt', 'r') as file:
            r2s = eval(file.read())


        X = [i  for _ in range(N)]
        Y = r2s
        YT = [[row[i] for row in Y] for i in range(len(Y[0]))]
        for m in [0,1,2]:
            ax1.scatter(X, YT[m], c=color_names[m], alpha=0.75, s=size)

        # plt.grid(axis="x",c='r',linestyle='-.')

        if flag == 0:
            plt.legend(['baseline', 'proposed', 'task-bagging'])
            flag = 1

        for m in [0,1,2]:
            plt.plot([i - 0.025, i + 0.025], [np.median(YT[m]), np.median(YT[m])], c=color_names[m], linestyle='--',
                     linewidth=size / 3.5)

    plt.savefig("trailCtest_fig.png")
    plt.show()

Metrics_Dict_Names = {
    1:"JS_divergence",
    2:"MMD",
    3:"HSIC",
    4:"A-distance"
}

if generate_picture_3:# C值-集成策略 实验

    fig = plt.figure(dpi=300)
    ax1 = fig.add_subplot(111)
    size = 3
    plt.xlabel('metrics')
    plt.ylabel('R-square')
    plt.ylim(-1, 1)
    plt.xlim(0.5, 4.5)
    method_num = 4
    plt.xticks([])
    color_names = ['black','black', 'red', 'purple','green']

    flag = 0
    for i in Metrics_Dict_Names.keys():

        with open('dataMtest/data_r2s_' + str(i-1) + '.txt', 'r') as file:
            r2s = eval(file.read())

        X = [_ / N / 2 + 0.75 + i - 1 for _ in range(N)]
        X = [i + _/N/10 for _ in range(N)]
        Y = r2s
        YT = [[row[i] for row in Y] for i in range(len(Y[0]))]
        for m in [1]:
            ax1.scatter(X, YT[m], c=color_names[i], alpha=0.75, s=size,label=Metrics_Dict_Names[i])

        # plt.grid(axis="x",c='r',linestyle='-.')
        flag += 100

        if flag == 1:
            plt.legend(["JS_divergence"])
        if flag == 2:
            plt.legend(["MMD"])
        if flag == 3:
            plt.legend(["HSIC"])
        if flag == 4:
            plt.legend(["A-distance"])

        plt.legend()

        for m in [1]:
            plt.plot([i - 0.25, i + 0.25], [np.median(YT[m]), np.median(YT[m])], c=color_names[i], linestyle='--',
                     linewidth=size / 3.5)

    plt.savefig("trailMtest_fig.png")
    plt.show()

if generate_picture_4:# 噪声实验

    fig = plt.figure(dpi=300)
    ax1 = fig.add_subplot(111)
    size = 3
    plt.xlabel('value of c')
    plt.ylabel('R-square')
    plt.ylim(-1, 1)
    plt.xlim(-0.4, 1)
    plt.xticks([K/20 for K in range(-6,18,2)])
    method_num = 1
    color_names = ['black', 'red', 'purple']

    flag = 0
    for i in [K/20 for K in range(-6,18,2)]:

        with open('dataCtest_wm/data_r2s_' + str(i) + '.txt', 'r') as file:
            r2s = eval(file.read())


        X = [i  for _ in range(N)]
        Y = r2s
        YT = [[row[i] for row in Y] for i in range(len(Y[0]))]
        for m in [0,1,2]:
            ax1.scatter(X, YT[m], c=color_names[m], alpha=0.75, s=size)

        # plt.grid(axis="x",c='r',linestyle='-.')

        if flag == 0:
            plt.legend(['baseline', 'proposed', 'task-bagging'])
            flag = 1

        for m in [0,1,2]:
            plt.plot([i - 0.025, i + 0.025], [np.median(YT[m]), np.median(YT[m])], c=color_names[m], linestyle='--',
                     linewidth=size / 3.5)

    plt.savefig("trailCtestwm_fig.png")
    plt.show()