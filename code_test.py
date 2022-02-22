import random
import pandas as pd
import numpy as np
import sys

from data_prepare import make_data,make_data_2th
from algo_utils import *
from sklearn.linear_model import LinearRegression

if 0:
    print(".1")
    test_task_rows = 700
    l = []
    for i in range(1,10):
        l.append(make_data(1,test_task_rows,alphas=[i]))
    test = make_data(1,test_task_rows,alphas=[7])

    for i in range(1,9):
        a = l[i].copy()
        b = test.copy()
        a = a.reset_index(drop=True)
        b = b.reset_index(drop=True)
        msg = muti_js_micro(l[i],test)
        print(msg)

    print("end")


if 0:
    print( 5 / 2)
    from numpy import e
    print(e ** 1)
    print("code test")
    
    from data_prepare import make_data
    from algo_utils import *
    ds = []
    for i in range(10):
        ds.append(make_data(1,1000,alphas=[i+1]))
    
    dtt = make_data(10,1000)
    dt = make_data(1,200,alphas=[4.5])
    m = prediction_pipeline(dtt,ProModel,task_rows=1000)
    prediction_pipeline(dt,fitted=m,output="y_hat")

if 0:

    dtrain = make_data(10,1000)
    dtest = make_data(1,1000,alphas=[3])
    d = dtrain
    Lx,Ly = ProModel.make_batch(dtrain,None,1000)
    """
    for i in range(len(Lx)):
        c = Lx[i]
        Lx[i] = c.reset_index(drop=True)
    """
    d = dtest
    Yt = d["Y"]
    Xt = d.drop(["Y"], axis=1)
    print(len(Lx))
    print(type(Lx))
    print(type(Lx[0]))
    print(Lx[0].shape)
    print(Lx[0].columns.tolist())

    print(len(Xt))
    print(type(Xt))
    print(Xt.shape)


    print("?" * 20)

    print(muti_js_micro(dtrain[0:1000].drop(["Y"], axis=1).reset_index(drop=True), Xt))
    print(muti_js_micro(dtrain[1000:2000].drop(["Y"], axis=1).copy(), Xt))
    print(muti_js_micro(dtrain[1000:2000].drop(["Y"], axis=1).reset_index(drop=True), Xt))

    print(".2")

    print(muti_js_micro(Lx[0],Xt))
    print(muti_js_micro(Lx[1],Xt))
    print(muti_js_micro(Lx[2],Xt))
    print(muti_js_micro(Lx[3],Xt))
    print(muti_js_micro(Lx[4],Xt))
    print(muti_js_micro(Lx[5],Xt))
    print(muti_js_micro(Lx[6],Xt))
    print(muti_js_micro(Lx[7],Xt))
    print(muti_js_micro(Lx[8],Xt))
    print(muti_js_micro(Lx[9],Xt))

    print(".3")

    Xt = Lx[2].copy()
    print(muti_js_micro(Lx[0], Xt))
    print(muti_js_micro(Lx[1], Xt))
    print(muti_js_micro(Lx[2], Xt))
    print(muti_js_micro(Lx[3], Xt))
    print(muti_js_micro(Lx[4], Xt))
    print(muti_js_micro(Lx[5], Xt))
    print(muti_js_micro(Lx[6], Xt))
    print(muti_js_micro(Lx[7], Xt))
    print(muti_js_micro(Lx[8], Xt))
    print(muti_js_micro(Lx[9], Xt))

    print(".4")

    Xt = make_data(1,1000,alphas=[3])
    Xt_2 = make_data(1, 1000, alphas=[1000])
    Xt = Xt.drop(["Y"], axis=1)
    Xt_2 = d.drop(["Y"], axis=1)
    Xt = Xt.reset_index(drop = True)
    Xt_2.reset_index(drop=True)


    print(muti_js_micro(Lx[0], Xt))
    print(muti_js_micro(Lx[1], Xt))
    print(muti_js_micro(Lx[2], Xt))
    print(muti_js_micro(Lx[3], Xt))
    print(muti_js_micro(Lx[4], Xt))
    print(muti_js_micro(Lx[5], Xt))
    print(muti_js_micro(Lx[6], Xt))
    print(muti_js_micro(Lx[7], Xt))
    print(muti_js_micro(Lx[8], Xt))
    print(muti_js_micro(Lx[9], Xt))

    print(Xt)
    print(Lx[9])


if 0:
    Xnum = 5
    Xnames = ["X" + str(i) for i in range(1, 1 + Xnum)]
    print(Xnames)

    for i in range(10):
        mu = np.random.uniform(0,5)
        print(mu)

    print(6 // 2 * 7)
    sys.exit()

if 1:
    dtrain = make_data_2th(15, 1000, data_scene=[8,8,0,0,0,0,0],U_num=5)
    #dtest = make_data_2th(1, 1000, data_scene=[2,20,2,6,1,2,1])
    print(dtrain.columns)
    #print(dtest.columns)
    print(dtrain)


