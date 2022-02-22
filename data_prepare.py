import random
import numpy as np
import pandas as pd
import logging

funcs = []
docs = {}
def management(f):
    funcs.append(f)
    f.__doc__ = docs[f.__name__]
    return f

docs["make_data"] = """
Make a data for transfer learning simulation.
Data is supposed to slightly change according to task-id. 

Parameters
----------
task_num : int, default = 1
    Number of tasks.
task_rows : int, default = 1000
    Create how many entries for each task.
alphas: list, default = None
    Params for data generation.

Returns
-------
m : pd.Dataframe
"""
docs["make_data_2th"] = docs["make_data"]

@management
def make_data(task_num=1,task_rows=1000,alphas=None,data_pattern=None): # disused
    # disused
    _ex_testuse_offset = 0 # should be 0
    data_pattern = 1 # 数据生成模式
    task_sensitive = 0.5
    total_rows = task_num * task_rows
    if alphas is None:
        alphas = list(range(1+_ex_testuse_offset,1+task_num+_ex_testuse_offset))
    else:
        pass
        # print(alphas)

    t = np.ones([total_rows, 6], dtype=float)
    d = pd.DataFrame(t)
    d.columns = ["Y", "X1", "X2", "Z1", "Z2", "taskalpha"]
    d["X1"] = np.random.normal(0, 1, total_rows)
    d["X2"] = np.random.normal(1, 1, total_rows)
    d["Y"] = np.empty(total_rows)


    for i in range(task_num):
        offset = i * (task_rows)
        start, end = offset, offset + (task_rows)
        alpha = alphas[i]
        p1, p2 = task_sensitive * alpha, 10 - task_sensitive * alpha
        noise = np.random.normal(0, 0.2, task_rows)
        #TODO 噪声影响确保小于父节点的
        if data_pattern == 1:
            d["Z1"][start:end] = np.random.normal(p1, 0.5, task_rows)
            d["Z2"][start:end] = np.random.normal(p2, 0.5, task_rows)
            d["Y"][start:end] = p1 * d["X1"][start:end] + p2 * 1 * d["X2"][start:end] + \
                                + p1 * d["Z1"][start:end]\
                                + p2 * d["Z2"][start:end] +noise

        if data_pattern == 2:
            d["Y"][start:end] = p1 * d["X1"][start:end] + p2 * 1 * d["X2"][start:end] + noise
            d["Z1"][start:end] = p1 * d["Y"][start:end] + np.random.normal(p1, 0.5, task_rows)
            d["Z2"][start:end] = p2 * d["Y"][start:end] + np.random.normal(p2, 0.5, task_rows)
            d["taskalpha"][start:end] = alpha

        if data_pattern == 3:
            d["Y"][start:end] = p1 * d["X1"][start:end] + p2 * 1 * d["X2"][start:end] + noise
            d["Z1"][start:end] = p1 * d["Y"][start:end] + np.random.normal(p1, 0.5, task_rows) + d["X1"][start:end]
            d["Z2"][start:end] = p2 * d["Y"][start:end] + np.random.normal(p2, 0.5, task_rows) + d["X2"][start:end]
            d["taskalpha"][start:end] = alpha

        if data_pattern == 4:
            d["Y"][start:end] = p1 * d["X1"][start:end] + p2 * 1 * d["X2"][start:end] + noise
            d["Z1"][start:end] = np.random.normal(p1, 0.5, task_rows)
            d["Z2"][start:end] = np.random.normal(p2, 0.5, task_rows)
            d["taskalpha"][start:end] = alpha

        if data_pattern == 5:
            d["Y"][start:end] = 5 * d["X1"][start:end] + 5 * 1 * d["X2"][start:end] + noise
            d["Z1"][start:end] = d["Y"][start:end] + np.random.normal(0, 0.5, task_rows)
            d["Z2"][start:end] = d["Y"][start:end] + np.random.normal(0, 0.5, task_rows)
            d["taskalpha"][start:end] = alpha

        d["taskalpha"][start:end] = alpha

    d = d.drop(["taskalpha"], axis=1)
    return d

@management
def make_data_2th(task_num=1,task_rows=1000,alphas=None,data_scene=None,seed=0,U_num=0,**kwargs):
    np.random.seed(seed)
    random.seed(seed)
    _ex_testuse_offset = 0 # should be 0 # disused
    data_pattern = 1 # 数据生成模式 # disused
    task_sensitive = 0.5 # disused
    coeff_upperbound = 3
    coeff_lowerbound = -3
    var_upperbound = 5
    var_lowerbound = 0
    y_noise_var = 0.05
    z_noise_var = 0.05
    _deltas = [np.random.normal(0.5,0.5) for _ in range(task_num)]
    if kwargs:
        if "coeff_upperbound" in kwargs:
            coeff_upperbound = kwargs["coeff_upperbound"]
        if "coeff_lowerbound" in kwargs:
            coeff_lowerbound = kwargs["coeff_lowerbound"]
        if "var_upperbound" in kwargs:
            var_upperbound = kwargs["var_upperbound"]
        if "var_lowerbound" in kwargs:
            var_lowerbound = kwargs["var_lowerbound"]
            print("debug :",str(var_lowerbound))


    total_rows = task_num * task_rows
    if data_scene is None:
        data_scene = [8,8,0,0,0,0,0]

    p = sum(data_scene) + U_num

    if alphas is None: # disused
        alphas = list(range(1+_ex_testuse_offset,1+task_num+_ex_testuse_offset))
    else:
        pass
        # print(alphas)

    t = np.ones([total_rows, p+2], dtype=float)
    d = pd.DataFrame(t)
    columns_name = []

    assert data_scene[0] % 2 == 0
    assert data_scene[1] % 2 == 0
    assert data_scene[2] % 2 == 0
    assert data_scene[3] % 6 == 0
    assert data_scene[4] % 1 == 0
    assert data_scene[5] % 2 == 0
    assert data_scene[6] % 1 == 0

    globalXnum = (data_scene[0] + data_scene[1]) // 2
    specialXnum = (data_scene[0] + data_scene[1]) // 2
    globalZnum = data_scene[2] // 2
    specialZnum = data_scene[3] // 2
    globalPnum = data_scene[2] // 2 + data_scene[3] // 6 * 2
    specialPnum = data_scene[3] // 6
    incompleteZnum = data_scene[4] + data_scene[5] + data_scene[6]
    # ======
    Xglobalnames = ["XG" + str(i) for i in range(1,1 + globalXnum)]
    Xspecialnames = ["XS" + str(i) for i in range(1, 1 + specialXnum)]
    # ======
    Zglobalnames = ["ZG" + str(i) for i in range(1, 1 + globalZnum)]
    Zspecialnames = ["ZS" + str(i) for i in range(1, 1 + specialZnum)]
    # ======
    Pglobalnames = ["PG" + str(i) for i in range(1, 1 + globalPnum)]
    Pspecialnames = ["PS" + str(i) for i in range(1, 1 + specialPnum)]
    # ======
    Zincompletenames = ["ZI" + str(i) for i in range(1, 1 + incompleteZnum)]
    # ======
    Unames = ["U" + str(i) for i in range(1, 1 + U_num)]
    # G:全局 S：环境特定 I：缺省父节点
    # 对于X，蓝色是S无色是G。对于Z和G，红色是S蓝色无色是G。

    columns_name = ["Y"] + Xglobalnames + Xspecialnames + Zglobalnames + \
                   Zspecialnames + Zincompletenames + Pglobalnames + \
                   Pspecialnames + Unames + ["taskalpha"]
    d.columns = columns_name
    print("debug :",str(columns_name))

    data_scene[0] = 0

    d["Y"] = np.zeros(total_rows)

    # data init part-1 XG
    Xindex = 1
    for j in range(globalXnum):
        mu = np.random.uniform(var_lowerbound, var_upperbound)
        d["XG"+str(Xindex)] = np.random.normal(mu, 1, total_rows)
        Xindex += 1

    # data init part-2 XS
    Xindex = 1
    for j in range(specialXnum):
        for i in range(task_num):
            offset = i * (task_rows)
            start, end = offset, offset + (task_rows)
            mu = np.random.uniform(var_lowerbound, var_upperbound)
            d["XS" + str(Xindex)][start:end] = np.random.normal(mu + _deltas[i], 1, task_rows)
        Xindex += 1

    # data init part-3 Y
    global_beta_num = (data_scene[0] + data_scene[1]) // 2
    special_beta_num = (data_scene[0] + data_scene[1]) // 2
    global_betas = [np.random.uniform(coeff_lowerbound, coeff_upperbound) for _ in range(global_beta_num)]
    for i in range(task_num):
        offset = i * (task_rows)
        start, end = offset, offset + (task_rows)
        noise = np.random.normal(0, y_noise_var, task_rows)
        Xindex = 1
        for j in range(data_scene[0]//2):
            d["Y"][start:end] += d["XG" + str(Xindex)][start:end] * global_betas[2 * Xindex - 2]
            d["Y"][start:end] += d["XS" + str(Xindex)][start:end] * global_betas[2 * Xindex - 1]
            Xindex += 1
        for j in range(data_scene[1]//2):
            d["Y"][start:end] += d["XG" + str(Xindex)][start:end] * \
                                 np.random.uniform(coeff_lowerbound + _deltas[i],coeff_upperbound + _deltas[i])
            d["Y"][start:end] += d["XS" + str(Xindex)][start:end] * \
                                 np.random.uniform(coeff_lowerbound + _deltas[i],coeff_upperbound + _deltas[i])
            Xindex += 1
        d["Y"][start:end] += noise
        d["taskalpha"][start:end] = i + 1

    # data init part-4 Z with P
    global_mu_num_forP = data_scene[2] // 2 + data_scene[3] // 6 * 2 # we give extra mu forP to keep codes neat
    global_mus_forP = [np.random.uniform(var_lowerbound, var_upperbound) for _ in range(global_mu_num_forP)]
    global_gamma_num = data_scene[2] // 2 + data_scene[3] // 3
    global_gammas = [np.random.uniform(coeff_lowerbound, coeff_upperbound) for _ in range(global_gamma_num)]
    global_theta_num = data_scene[2] // 2 + data_scene[3] // 3
    global_thetas = [np.random.uniform(coeff_lowerbound, coeff_upperbound) for _ in range(global_theta_num)]
    for i in range(task_num):
        offset = i * (task_rows)
        start, end = offset, offset + (task_rows)
        Zindex = 1
        PGindex = 1
        PSindex = 1
        gammaindex = 0
        thetaindex = 0
        for j in range(data_scene[2] // 2): # P1 Z1
            d["PG" + str(PGindex)][start:end] = np.random.normal(global_mus_forP[PGindex - 1], 1, task_rows)
            d["ZG" + str(Zindex)][start:end] = d["Y"][start:end] * global_gammas[gammaindex]+ \
                                               d["PG" + str(PGindex)][start:end] * global_thetas[thetaindex] + \
                                               np.random.normal(0, z_noise_var, task_rows)

            gammaindex += 1
            thetaindex += 1
            PGindex += 1
            Zindex += 1

        Zindex = 1
        for j in range(data_scene[3] // 6): # P2 Z2
            mu = np.random.uniform(var_lowerbound, var_upperbound)
            d["PS" + str(PSindex)][start:end] = np.random.normal(mu + _deltas[i], 1, task_rows)
            d["ZS" + str(Zindex)][start:end] = d["Y"][start:end] * global_gammas[gammaindex] + \
                                               d["PS" + str(PSindex)][start:end] * global_thetas[thetaindex] + \
                                               np.random.normal(0, z_noise_var, task_rows)
            gammaindex += 1
            thetaindex += 1
            PSindex += 1
            Zindex += 1

        for j in range(data_scene[3] // 6): # P3 Z3
            d["PG" + str(PGindex)][start:end] = np.random.normal(global_mus_forP[PGindex - 1], 1, task_rows)
            d["ZS" + str(Zindex)][start:end] = d["Y"][start:end] * global_gammas[gammaindex] + \
                                               d["PG" + str(PGindex)][start:end] * \
                                               np.random.uniform(coeff_lowerbound + _deltas[i],coeff_upperbound + _deltas[i]) + \
                                               np.random.normal(0, z_noise_var, task_rows)
            gammaindex += 1
            PGindex += 1
            Zindex += 1

        for j in range(data_scene[3] // 6): # P4 Z4
            d["PG" + str(PGindex)][start:end] = np.random.normal(global_mus_forP[PGindex - 1], 1, task_rows)
            d["ZS" + str(Zindex)][start:end] = d["Y"][start:end] * \
                                               np.random.uniform(coeff_lowerbound + _deltas[i],coeff_upperbound + _deltas[i]) + \
                                               d["PG" + str(PGindex)][start:end] * global_thetas[thetaindex] + \
                                               np.random.normal(0, z_noise_var, task_rows)
            thetaindex += 1
            PGindex += 1
            Zindex += 1

    # data init part-5 Z missing P
    global_mu_num_forP = data_scene[4] + data_scene[5] // 2 + data_scene[6]
    global_mus_forP = [np.random.uniform(var_lowerbound, var_upperbound) for _ in range(global_mu_num_forP)]
    global_gamma_num = data_scene[4] + data_scene[5]
    global_gammas = [np.random.uniform(coeff_lowerbound, coeff_upperbound) for _ in range(global_gamma_num)]
    global_theta_num = data_scene[4] + data_scene[5] // 2 + data_scene[6]
    global_thetas = [np.random.uniform(coeff_lowerbound, coeff_upperbound) for _ in range(global_theta_num)]
    for i in range(task_num):
        offset = i * (task_rows)
        start, end = offset, offset + (task_rows)
        Zindex = 1
        Pindex = 1
        gammaindex = 0
        thetaindex = 0
        for j in range(data_scene[4]):  # P1 Z1
            tempP = np.random.normal(global_mus_forP[Pindex - 1], 1, task_rows)
            d["ZI" + str(Zindex)][start:end] = d["Y"][start:end] * global_gammas[gammaindex] + \
                                               tempP * global_thetas[thetaindex] + \
                                               np.random.normal(0, z_noise_var, task_rows)
            gammaindex += 1
            thetaindex += 1
            Pindex += 1
            Zindex += 1

        for j in range(data_scene[5]//2):  # P2 Z2
            mu = np.random.uniform(var_lowerbound, var_upperbound)
            tempP = np.random.normal(mu + _deltas[i], 1, task_rows)
            d["ZI" + str(Zindex)][start:end] = d["Y"][start:end] * global_gammas[gammaindex] + \
                                               tempP * global_thetas[thetaindex] + \
                                               np.random.normal(0, z_noise_var, task_rows)
            gammaindex += 1
            thetaindex += 1 # Pindex
            Zindex += 1

        for j in range(data_scene[5]//2):  # P3 Z3
            tempP = np.random.normal(global_mus_forP[Pindex - 1], 1, task_rows)
            d["ZI" + str(Zindex)][start:end] = d["Y"][start:end] * global_gammas[gammaindex] + \
                                               tempP * np.random.uniform(coeff_lowerbound + _deltas[i], coeff_upperbound + _deltas[i]) + \
                                               np.random.normal(0, z_noise_var, task_rows)
            gammaindex += 1
            Pindex += 1
            Zindex += 1

        for j in range(data_scene[6]):  # P4 Z4
            tempP = np.random.normal(global_mus_forP[Pindex - 1], 1, task_rows)
            d["ZI" + str(Zindex)][start:end] = d["Y"][start:end] * \
                                               np.random.uniform(coeff_lowerbound + _deltas[i], coeff_upperbound + _deltas[i]) + \
                                               tempP * global_thetas[thetaindex] + \
                                               np.random.normal(0, z_noise_var, task_rows)
            thetaindex += 1
            Pindex += 1
            Zindex += 1

    # data init part-5 U
    Uindex = 1
    for j in range(U_num):
        d["U" + str(Uindex)] = np.random.normal(0, 1, total_rows)
        Uindex += 1

    d = d.drop(["taskalpha"], axis=1)
    return d


"""
print(make_data.__doc__)
"""