import random
import sys

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

from data_prepare import make_data,make_data_2th
from algo_utils import *


np.random.seed(0)
random.seed(0)
# global params
#
train_task_rows = 2000
test_task_rows = 2000

true_C = 5 # 测试任务的 task id
base_learner = DecisionTreeRegressor # 使用何种基学习器
# make data
train = make_data(9,train_task_rows)
test = make_data(1,test_task_rows,alphas=[true_C])

train = make_data_2th(10, 2000, data_scene=[4,4,4,12,1,2,1])
test = make_data_2th(1, 2000, data_scene=[4,4,4,12,1,2,1])

# OLS linear regression
print(" *** baseline 全局模型 *** ")
# 训练模型
model = prediction_pipeline(train,base_learner)

# 测试模型
_,mse = prediction_pipeline(test,fitted=model,output="y_hat")

print("MSE")
print(mse)
if base_learner == LinearRegression:
    print("系数：",model.coef_)
    print("截距项：",model.intercept_)
basemse = mse

# 本文提出的方法
print("=" * 88)
print(" *** 本文方法 新模型 *** ")
# 训练模型
newmodel = prediction_pipeline(train,ProModel,task_rows=train_task_rows,
                               base_learner = base_learner,distance_metrics = muti_js_micro,verbose = True)

# 测试模型
r,mse = prediction_pipeline(test,fitted=newmodel,output="y_hat")
mse_2 = mse

print("MSE")
print(mse)
print("_scores",newmodel._scores)
print("_task_num",newmodel.task_num)
print("base_mse",basemse)
if base_learner == LinearRegression:
    for i in range(9):
        print("第" + str(i) +"个任务：系数：",newmodel._base_learners[i].coef_)
print(newmodel._X[0].shape)
print(newmodel._X[0].columns)
print("=" * 88)

# 单个模型
print(" *** 单个模型 *** ")
model = prediction_pipeline(train[0000 + int(2000 * (true_C-1)):2000 + int(2000 * (true_C-1))],LinearRegression)
_,mse = prediction_pipeline(test,fitted=model,output="y_hat")
mse_3 = mse
print("MSE")
print(mse)
print("系数：",model.coef_)

print("/")
print("全局模型，本文模型，真实单任务模型的mse分别为:",basemse,",",mse_2,",",mse_3)
print("=" * 88)