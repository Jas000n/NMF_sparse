import numpy as np


def get_k_ratings(erm, rm,U,k):
    m = erm.shape[0]
    n = erm.shape[1]
    #print(matrix)
    row_U = erm[U, :]
    avarage = row_U.sum() / n
    row_actual = row_U.tolist()
    #print(row_U, avarage)
    row_U = row_U - avarage
    #print(row_i)
    arg = row_U.argsort().tolist()
    arg = arg[::-1]
    rated_list = get_rated_index(rm, U)
    #print(rated_list)
    for i in arg:
        if(i in rated_list):
            #print(i)
            arg.remove(i)
    arg = arg[0:k]
    print("Now recommend {} item for User {}, according to {} items he has rated, his average rating score is {}:".format(k,U,len(rated_list),avarage))
    for i in range(0,k):
        print("item id:{}\t\testimated rating for him:{}".format(arg[i]+1,row_actual[arg[i]]))



# return rated list of user U according to a rating matrix, in the form of index
def get_rated_index(matrix, U):
    row_u = matrix[U, :]
    arg_rated = row_u.argsort()
    rated_list = []
    for i in arg_rated:
        if (row_u[i] != 0):
            rated_list.append(i)
    return rated_list
