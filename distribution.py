import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from surprise import NMF
from surprise import Dataset

import mv100

mode = "clustered"
if mode == "general_mymodel":
    erm = np.load("save_matrix299.npy", allow_pickle=True)
    erm = erm.reshape((1, -1))

    rm = mv100.mv1002list("./ml-100k/u5_fix.base")
    rm = np.array(rm, dtype=int)
    rm = rm[:, 2]
    sns.distplot(rm, label="actual distribution")
    sns.distplot(erm, label="estimated distribution")
    plt.title("my model")
    plt.legend()
    plt.show()
elif mode == "random":
    while (True):
        u = eval(input("check density distribution on some random user:"))
        erm_sup = np.load("sup_model.npy", allow_pickle=True)
        erm_sup = erm_sup[u, :]
        erm_sup = erm_sup.reshape((1, -1))
        erm_mine = np.load("save_matrix299.npy", allow_pickle=True)
        erm_mine = erm_mine[u, :]
        erm_mine = erm_mine.reshape((1, -1))
        rm = mv100.mv1002list("./ml-100k/u5_fix.base")
        rm = np.array(rm, dtype=int)
        U_rm = []
        for i in rm:
            if (i[0] == u):
                U_rm.append(i[2])
        # print(len(U_rm))
        sns.distplot(erm_sup, label="estimated distribution from SUPRISE")
        sns.distplot(erm_mine, label="estimated distribution from My_Model")
        sns.distplot(U_rm, label="actual distribution")
        plt.legend()
        plt.title("{}'s rating density distribution".format(u))
        plt.show()
elif mode == "general_sup":
    erm_sup = np.load("sup_model.npy", allow_pickle=True)
    erm_sup = erm_sup.reshape((1, -1))

    rm = mv100.mv1002list("./ml-100k/u5_fix.base")
    rm = np.array(rm, dtype=int)
    rm = rm[:, 2]
    sns.distplot(rm, label="actual distribution")
    sns.distplot(erm_sup, label="estimated distribution")
    plt.title("Surprise NMF library")
    plt.legend()
    plt.show()
elif mode == "general_compare":
    erm_sup = np.load("sup_model.npy", allow_pickle=True)
    erm_sup = erm_sup.reshape((1, -1))
    erm_mine = np.load("save_matrix299.npy", allow_pickle=True)
    erm_mine = erm_mine.reshape((1, -1))

    rm = mv100.mv1002list("./ml-100k/u5_fix.base")
    rm = np.array(rm, dtype=int)
    rm = rm[:, 2]
    sns.distplot(rm, label="actual distribution")
    sns.distplot(erm_sup, label="estimated distribution_sup")
    sns.distplot(erm_mine, label="estimated distribution_mine")
    plt.title("Surprise vs mine")
    plt.legend()
    plt.show()
elif mode == "clustered":
    type1_actual = np.load("./clusters/type1.npy", allow_pickle=True)
    type2_actual = np.load("./clusters/type2.npy", allow_pickle=True)
    type1_actual_plt = []
    type2_actual_plt = []
    for i in type1_actual:
        for j in range(0, int(i[1] * 1586126 / 100)):
            type1_actual_plt.append(i[0])
    for i in type2_actual:
        for j in range(0, int(i[1] * 1586126 / 100)):
            type2_actual_plt.append(i[0])
    sns.distplot(type1_actual_plt, label="preferences of type1 people")
    sns.distplot(type2_actual_plt, label="preferences of type2 people")
    plt.title("Distribution of 2 group of people")
    plt.legend()
    plt.show()

elif mode == "type1&type2":
    type1_actual = np.load("./clusters/type1.npy", allow_pickle=True)
    type2_actual = np.load("./clusters/type2.npy", allow_pickle=True)

    type1_actual_plt=[]
    type2_actual_plt=[]
    for i in type1_actual:
        for j in range(0,int(i[1]* 1586126 / 100)):
            type1_actual_plt.append(i[0])
    for i in type2_actual:
        for j in range(0, int(i[1] * 1586126 / 100)):
            type2_actual_plt.append(i[0])
    sns.distplot(type1_actual_plt, label="preferences of type1 people")
    sns.distplot(type2_actual_plt, label="preferences of type2 people")
    type1_erm = np.load("./type1/save_matrix_type1_299.npy", allow_pickle=True)
    type1_erm = type1_erm.reshape((1, -1))
    type2_erm = np.load("./type2/save_matrix_type2_299.npy", allow_pickle=True)
    type2_erm = type2_erm.reshape((1, -1))
    sns.distplot(type1_erm, label="estimated for type1")
    sns.distplot(type2_erm, label="estimated for type2")
    plt.title("my model")
    plt.legend()
    plt.show()
