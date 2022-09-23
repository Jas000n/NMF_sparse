import torch


def RMSE(x,y):
    len = x.nonzero().size()[0]
    print(len)
    loss = ((x-y)*(x-y)).sum()/len
    loss = loss**0.5
    return  loss

x = torch.Tensor([[1,2,3,4,5],[0,0,0,0,0]])
y = torch.Tensor([[2,3,4,5,6],[0,0,0,0,0]])

print(RMSE(x,y))