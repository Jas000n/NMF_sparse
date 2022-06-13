from matplotlib import pyplot as plt

from plt import plot


with open("/Users/jas0n/PycharmProjects/NMF_sparse/logs/output1.out") as Type1:
    print(Type1)
    lines = Type1.readlines()
    i = 0
    x=[]
    train_loss_1=[]
    test_loss_1=[]
    for line in lines:

        splited_train = line.split("rmse loss on training set is")
        splited_test = line.split("rmse loss on test set is")
        if (len(splited_train) == 2):
            # rmse loss on train set
            print(splited_train[1])
            train_loss_1.append(float(splited_train[1].split("\n")[0].split(" ")[1]))
            x.append(i)
            i += 1
        if (len(splited_test) == 2):
            # rmse loss on test set
            print(splited_test[1])
            test_loss_1.append(float(splited_test[1].split("\n")[0].split(" ")[1]))
print(len(train_loss_1))
print(len(test_loss_1))


with open("/Users/jas0n/PycharmProjects/NMF_sparse/logs/output2.out") as Type2:
    print(Type2)
    lines = Type2.readlines()
    i = 0
    x=[]
    train_loss_2=[]
    test_loss_2=[]
    for line in lines:

        splited_train = line.split("rmse loss on training set is")
        splited_test = line.split("rmse loss on test set is")
        if (len(splited_train) == 2):
            # rmse loss on train set
            print(splited_train[1])
            train_loss_2.append(float(splited_train[1].split("\n")[0].split(" ")[1]))
            x.append(i)
            i += 1
        if (len(splited_test) == 2):
            # rmse loss on test set
            print(splited_test[1])
            test_loss_2.append(float(splited_test[1].split("\n")[0].split(" ")[1]))
print(len(train_loss_2))
print(len(test_loss_2))


with open("/Users/jas0n/PycharmProjects/NMF_sparse/logs/result.txt") as result:
    print(result)
    lines = result.readlines()
    i = 0
    x=[]
    train_loss_all=[]
    test_loss_all=[]
    for line in lines:

        splited_train = line.split("rmse loss on training set is")
        splited_test = line.split("rmse loss on test set is")
        if (len(splited_train) == 2):
            # rmse loss on train set
            print(splited_train[1])
            train_loss_all.append(float(splited_train[1].split("\n")[0].split(" ")[1]))
            x.append(i)
            i += 1
        if (len(splited_test) == 2):
            # rmse loss on test set
            print(splited_test[1])
            test_loss_all.append(float(splited_test[1].split("\n")[0].split(" ")[1]))
print(len(train_loss_all))
print(len(test_loss_all))
plt.plot(x,train_loss_1,label="train_loss on type1 user")
plt.plot(x,test_loss_1,label="test_loss on type1 user")
plt.plot(x,train_loss_2,label="train_loss on type2 user")
plt.plot(x,test_loss_2,label="test_loss on type2 user")
plt.plot(x,train_loss_all,label="train_loss using one model on all user")
plt.plot(x,test_loss_all,label="test_loss using one model on all user")
plt.legend()
plt.savefig("result_type1&2&all.png")
plt.show()