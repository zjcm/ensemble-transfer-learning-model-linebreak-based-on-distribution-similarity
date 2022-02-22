import random
import sys

import pandas as pd
import numpy as np


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

from data_prepare import make_data,make_data_2th
from algo_utils import *


np.random.seed(0)
random.seed(0)

# global params
#
train_task_rows = 2000
test_task_rows = 2000

base_learner = DecisionTreeRegressor # 使用何种基学习器
# make data

data_scenes = {
    1:[4,4,0,0,0,0,0],
    2:[4,4,4,12,0,0,0],
    3:[4,4,4,12,2,4,2],
    4:[0,0,4,12,0,0,0],
    5:[0,0,4,12,2,4,2],
    6:[0,0,0,0,2,4,2],
    7:[4,0,0,0,0,0,0],
    8:[0,4,0,0,0,0,0],
    9:[0,0,4,0,0,0,0],
    10:[0,0,0,12,0,0,0],
    11:[0,0,0,0,4,0,0],
    12:[0,0,0,0,0,4,0],
    13:[0,0,0,0,0,0,4]
    
}

results = []
dens = []
r2s = []
props = []
N = 30
tasknum = 12

def main_trail(): # 第一组实验，对比各个算法的预测效果
    for s in data_scenes:
        print(" *** 第 " + str(s) + " 个场景 ***")
        print("*" * 80)
        print()
        r2s = []
        for i in range(1, N + 1):
            print(" *** trail " + str(i) + " *** ")
            scene = data_scenes[s]
            train = make_data_2th(tasknum, 2000, data_scene=scene.copy(), seed=i,
                                  coeff_lowerbound=0.5, coeff_upperbound=1, var_lowerbound=0.5, var_upperbound=1)
            test = make_data_2th(1, 2000, data_scene=scene.copy(), seed=N + i,
                                 coeff_lowerbound=0.5, coeff_upperbound=1, var_lowerbound=0.5, var_upperbound=1)
            den = np.var(train["Y"])

            # OLS linear regression

            print(" *** baseline 全局模型 *** ")
            # 训练模型
            model = prediction_pipeline(train, base_learner)

            # 测试模型
            _, mse = prediction_pipeline(test, fitted=model, output="y_hat")
            basemse = mse

            # 本文提出的方法
            print(" *** 本文方法 新模型 *** ")
            # 训练模型
            newmodel = prediction_pipeline(train, ProModel, task_rows=train_task_rows,
                                           base_learner=base_learner, distance_metrics=muti_js_micro, verbose=True)

            # 测试模型
            r, mse = prediction_pipeline(test, fitted=newmodel, output="y_hat")
            mse_2 = mse

            """
            # 单个模型
            print(" *** 单个模型 *** ")
            true_C = 1
            model = prediction_pipeline(train[0000 + int(2000 * (true_C-1)):2000 + int(2000 * (true_C-1))],LinearRegression)
            _,mse = prediction_pipeline(test,fitted=model,output="y_hat")
            mse_3 = mse
            """

            # 全局随机森林 如启用，比较耗时
            if 1:
                print(" *** 全局随机森林 *** ")
                # 训练模型
                model = prediction_pipeline(train, RandomForestRegressor)

                # 测试模型
                _, mse = prediction_pipeline(test, fitted=model, output="y_hat")
                mse_3 = mse
            mse_3 = 0

            # 按环境bagging
            print(" *** 按环境bagging *** ")
            # 训练模型
            newmodel = prediction_pipeline(train, ProModel, task_rows=train_task_rows,
                                           base_learner=base_learner, distance_metrics=muti_js_micro, verbose=False,
                                           promodel_score="even")

            # 测试模型
            _, mse = prediction_pipeline(test, fitted=newmodel, output="y_hat")
            mse_4 = mse

            # 汇总
            results.append([basemse, mse_2, mse_3, mse_4])
            dens.append(den)
            r2s.append([1 - basemse / den, 1 - mse_2 / den, 1 - mse_3 / den, 1 - mse_4 / den])
            props.append([basemse / basemse, mse_2 / basemse, mse_3 / basemse, mse_4 / basemse])

        with open('data4/data' + str(s) + '.txt', 'w') as file:
            file.write(str(r2s))

def noise_trail(): # 噪声变量影响测试
    for s in range(16):
        print(" *** 噪声变量影响测试 ***")
        print(" *** 第 " + str(s) + " 个场景 ***")
        print("*" * 80)
        print()
        r2s = []
        for i in range(1, N + 1):
            print(" *** trail " + str(i) + " *** ")
            scene = [4,4,0,0,0,0,0]
            train = make_data_2th(tasknum, 2000, data_scene=scene.copy(), seed= i + s * N,
                                  coeff_lowerbound=0.5, coeff_upperbound=1, var_lowerbound=0.5, var_upperbound=1,
                                  U_num=s)
            test = make_data_2th(1, 2000, data_scene=scene.copy(), seed=N + i + s * N,
                                 coeff_lowerbound=0.5, coeff_upperbound=1, var_lowerbound=0.5, var_upperbound=1,
                                 U_num=s)
            den = np.var(train["Y"])

            # OLS linear regression

            print(" *** baseline 全局模型 *** ")
            # 训练模型
            model = prediction_pipeline(train, base_learner)

            # 测试模型
            _, mse = prediction_pipeline(test, fitted=model, output="y_hat")
            basemse = mse

            # 本文提出的方法
            print(" *** 本文方法 新模型 *** ")
            # 训练模型
            newmodel = prediction_pipeline(train, ProModel, task_rows=train_task_rows,
                                           base_learner=base_learner, distance_metrics=muti_js_micro, verbose=True)

            # 测试模型
            r, mse = prediction_pipeline(test, fitted=newmodel, output="y_hat")
            mse_2 = mse

            """
            # 单个模型
            print(" *** 单个模型 *** ")
            true_C = 1
            model = prediction_pipeline(train[0000 + int(2000 * (true_C-1)):2000 + int(2000 * (true_C-1))],LinearRegression)
            _,mse = prediction_pipeline(test,fitted=model,output="y_hat")
            mse_3 = mse
            """
            """

            # 全局随机森林
            if 0:
                print(" *** 全局随机森林 *** ")
                # 训练模型
                model = prediction_pipeline(train, RandomForestRegressor)

                # 测试模型
                _, mse = prediction_pipeline(test, fitted=model, output="y_hat")
                mse_3 = mse
            """
            mse_3 = 0

            # 按环境bagging
            print(" *** 按环境bagging *** ")
            # 训练模型
            newmodel = prediction_pipeline(train, ProModel, task_rows=train_task_rows,
                                           base_learner=base_learner, distance_metrics=muti_js_micro, verbose=False,
                                           promodel_score="even")

            # 测试模型
            _, mse = prediction_pipeline(test, fitted=newmodel, output="y_hat")
            mse_4 = mse

            # 汇总
            dens.append(den)
            r2s.append([1 - basemse / den, 1 - mse_2 / den, 1 - mse_4 / den])

        with open('dataUtest/data' + str(s) + '.txt', 'w') as file:
            file.write(str(r2s))
    return

def c_trail(): # 集成策略c测试
    for s in [K/20 for K in range(-6,19,2)]:
        print(" *** 噪声变量影响测试 ***")
        print(" *** 第 " + str(s) + " 个场景 ***")
        print("*" * 80)
        print()
        r2s = []
        scoredata = []
        for i in range(1, N + 1):
            print(" *** trail " + str(i) + " *** ")
            scene = [4,4,0,0,0,0,0]
            train = make_data_2th(tasknum, 2000, data_scene=scene.copy(), seed= i,
                                  coeff_lowerbound=0.5, coeff_upperbound=1, var_lowerbound=0.5, var_upperbound=1,
                                  U_num=0)
            test = make_data_2th(1, 2000, data_scene=scene.copy(), seed=N + i,
                                 coeff_lowerbound=0.5, coeff_upperbound=1, var_lowerbound=0.5, var_upperbound=1,
                                 U_num=0)
            den = np.var(train["Y"])

            print(" *** baseline 全局模型 *** ")
            # 训练模型
            model = prediction_pipeline(train, base_learner)

            # 测试模型
            _, mse = prediction_pipeline(test, fitted=model, output="y_hat")
            basemse = mse

            # 本文提出的方法
            print(" *** 本文方法 新模型 *** ")
            # 训练模型
            scores, newmodel = prediction_pipeline(train, ProModel, task_rows=train_task_rows,
                                           base_learner=base_learner, distance_metrics=muti_js_micro, verbose=True,
                                           promodel_c=float(s),ask_scores=True)

            # 测试模型
            r, mse = prediction_pipeline(test, fitted=newmodel, output="y_hat")
            mse_2 = mse

            # 按环境bagging
            print(" *** 按环境bagging *** ")
            # 训练模型
            newmodel = prediction_pipeline(train, ProModel, task_rows=train_task_rows,
                                           base_learner=base_learner, distance_metrics=muti_js_micro, verbose=False,
                                           promodel_score="even")

            # 测试模型
            _, mse = prediction_pipeline(test, fitted=newmodel, output="y_hat")
            mse_4 = mse

            # 汇总
            dens.append(den)
            r2s.append([1 - basemse / den, 1 - mse_2 / den, 1 - mse_4 / den])
            scoredata.append(scores)

        with open('dataCtest/data_r2s_' + str(s) + '.txt', 'w') as file:
            file.write(str(r2s))
        with open('dataCtest/data_score_' + str(s) + '.txt', 'w') as file:
            file.write(str(scores))
    return

def c_trail_withA(): # 集成策略c测试
    for s in [K/20 for K in range(-6,18,2)]:
        print(" *** 噪声变量影响测试 ***")
        print(" *** 第 " + str(s) + " 个场景 ***")
        print("*" * 80)
        print()
        r2s = []
        scoredata = []
        for i in range(1, N + 1):
            print(" *** trail " + str(i) + " *** ")
            scene = [4,4,0,0,0,0,0]
            train = make_data_2th(tasknum, 2000, data_scene=scene.copy(), seed= i,
                                  coeff_lowerbound=0.5, coeff_upperbound=1, var_lowerbound=0.5, var_upperbound=1,
                                  U_num=0)
            test = make_data_2th(1, 2000, data_scene=scene.copy(), seed=N + i,
                                 coeff_lowerbound=0.5, coeff_upperbound=1, var_lowerbound=0.5, var_upperbound=1,
                                 U_num=0)
            den = np.var(train["Y"])

            print(" *** baseline 全局模型 *** ")
            # 训练模型
            model = prediction_pipeline(train, base_learner)

            # 测试模型
            _, mse = prediction_pipeline(test, fitted=model, output="y_hat")
            basemse = mse

            # 本文提出的方法
            print(" *** 本文方法 新模型 *** ")
            # 训练模型
            scores, newmodel = prediction_pipeline(train, ProModel, task_rows=train_task_rows,
                                           base_learner=base_learner, distance_metrics=A_distance, verbose=True,
                                           promodel_c=float(s),ask_scores=True)

            # 测试模型
            r, mse = prediction_pipeline(test, fitted=newmodel, output="y_hat")
            mse_2 = mse

            # 按环境bagging
            print(" *** 按环境bagging *** ")
            # 训练模型
            newmodel = prediction_pipeline(train, ProModel, task_rows=train_task_rows,
                                           base_learner=base_learner, distance_metrics=mmd_rbf, verbose=False,
                                           promodel_score="even")

            # 测试模型
            _, mse = prediction_pipeline(test, fitted=newmodel, output="y_hat")
            mse_4 = mse

            # 汇总
            dens.append(den)
            r2s.append([1 - basemse / den, 1 - mse_2 / den, 1 - mse_4 / den])
            scoredata.append(scores)

        with open('dataCtest_wm/data_r2s_' + str(s) + '.txt', 'w') as file:
            file.write(str(r2s))
        with open('dataCtest_wm/data_score_' + str(s) + '.txt', 'w') as file:
            file.write(str(scores))
    return

Metrics_Dict = {
    0:muti_js_micro,
    1:mmd_rbf,
    2:hsic_gam,
    3:A_distance
}

def M_trail(): # 相似性度量准则测试
    for M in [0,1,2,3]:
        print(" *** 相似性度量影响测试 ***")
        print(" *** 第 " + str(M) + " 个相似性度量 ***")
        print("*" * 80)
        print()
        r2s = []
        scoredata = []
        for i in range(1, N + 1):
            print(" *** trail " + str(i) + " *** ")
            scene = [4,4,0,0,0,0,0]
            train = make_data_2th(tasknum, 2000, data_scene=scene.copy(), seed= i,
                                  coeff_lowerbound=0.5, coeff_upperbound=1, var_lowerbound=0.5, var_upperbound=1,
                                  U_num=0)
            test = make_data_2th(1, 2000, data_scene=scene.copy(), seed=N + i,
                                 coeff_lowerbound=0.5, coeff_upperbound=1, var_lowerbound=0.5, var_upperbound=1,
                                 U_num=0)
            den = np.var(train["Y"])

            print(" *** baseline 全局模型 *** ")
            # 训练模型
            model = prediction_pipeline(train, base_learner)

            # 测试模型
            _, mse = prediction_pipeline(test, fitted=model, output="y_hat")
            basemse = mse

            # 本文提出的方法
            print(" *** 本文方法 新模型 *** ")
            # 训练模型
            _metrics = Metrics_Dict[M]
            newmodel = prediction_pipeline(train, ProModel, task_rows=train_task_rows,base_learner=base_learner,
                                              distance_metrics=_metrics,
                                              verbose=True)

            # 测试模型
            r, mse = prediction_pipeline(test, fitted=newmodel, output="y_hat")
            mse_2 = mse

            # 按环境bagging
            print(" *** 按环境bagging *** ")
            # 训练模型
            newmodel = prediction_pipeline(train, ProModel, task_rows=train_task_rows,
                                           base_learner=base_learner, distance_metrics=muti_js_micro, verbose=False,
                                           promodel_score="even")

            # 测试模型
            _, mse = prediction_pipeline(test, fitted=newmodel, output="y_hat")
            mse_4 = mse




            # 汇总
            dens.append(den)
            r2s.append([1 - basemse / den, 1 - mse_2 / den, 1 - mse_4 / den])

        with open('dataMtest/data_r2s_' + str(M) + '.txt', 'w') as file:
            file.write(str(r2s))

    return

if __name__ == '__main__':
    c_trail()

