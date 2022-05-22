import numpy as np
import id2MVname

def get_k_ratings(erm, rm,U,k):
    m = erm.shape[0]
    n = erm.shape[1]
    #print(matrix)
    row_U = erm[U, :]
    avarage = getAvarage(rm,U)
    row_estimated = row_U.tolist()
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
    print("Now recommend {} item for User {}, according to {} items he has rated, his average rating score is {}:".format(k,U+1,len(rated_list),round(avarage,3)))
    mvdic = id2MVname.getMvDict()
    for i in range(0,k):
        print("item id:{}\t\testimated rating for him:{}\n"
              "movie name is {}".format(arg[i]+1,round(row_estimated[arg[i]],3),mvdic.get(arg[i]+1)))



# return rated list of user U according to a rating matrix, in the form of index
def get_rated_index(matrix, U):
    row_u = matrix[U, :]
    arg_rated = row_u.argsort()
    rated_list = []
    for i in arg_rated:
        if (row_u[i] != 0):
            rated_list.append(i)
    return rated_list

def getAvarage(rm,U):
    row_toCalculate = rm[U,:]
    sum = 0
    rated_num = 0
    for i in row_toCalculate:
        if i !=0:
            sum+=i
            rated_num+=1
    return sum/rated_num