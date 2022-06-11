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
    type1 = np.load("./clusters/type1.npy", allow_pickle=True)
    type2 = np.load("./clusters/type2.npy", allow_pickle=True)

    sns.distplot(type1, label="preferences of type1 people")
    sns.distplot(type2, label="preferences of type2 people")
    plt.title("my model")
    plt.legend()
    plt.show()