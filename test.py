import torch
from torch import nn
from torch.utils.data import DataLoader
import plt
def RMSE(x,y):
    len = x.nonzero().size()[0]
    loss = ((x-y)*(x-y)).sum()/len
    loss = loss**0.5
    return  loss

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.netU = nn.Sequential(
            nn.Conv2d(1,1,(1,1682),1,0),

            # nn.Flatten(),
            # nn.Linear(943,943)

        )
        self.netI = nn.Sequential(
            nn.Conv2d(1,1,(943,1),1,0),

            # nn.Flatten(),
            # nn.Linear(1682,1682)
        )

    def forward(self, x):
        vectorU = self.netU(x).resize(943,1)
        vectorV = self.netI(x).resize(1,1682)
        return vectorU.mm(vectorV)

import mv100
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


train_list = mv100.mv1002list("./ml-100k/u5_fix.base")
test_list = mv100.mv1002list("./ml-100k/u5.test")
test_rm = torch.tensor(mv100.creat_matrix(test_list),dtype=torch.float).to(device)
train_rm = torch.tensor(mv100.creat_matrix(train_list),dtype=torch.float).to(device)
train_mask = train_rm.nonzero()
train_mask_ = torch.zeros((943,1682))
train_mask_ = train_mask_.to(device)

test_mask = test_rm.nonzero()
test_mask_ = torch.zeros((943,1682))
test_mask_ = test_mask_.to(device)
#得到zero mask
for i in train_mask:
    train_mask_[i[0]][i[1]] = 1
for i in test_mask:
    test_mask_[i[0]][i[1]] = 1

model = NeuralNetwork()
X = torch.rand((1,943,1682))
loss_fn = RMSE
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
model.to(device)
x =[]
test_loss__=[]
train_loss__=[]
for epoch in range(1,1000000):
    X, train_rm = X.to(device), train_rm.to(device)
    pred = model(X)
    pred_after_mask = pred*train_mask_
    test_loss = RMSE(pred*test_mask_,test_rm)
    loss = loss_fn(pred_after_mask, train_rm)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss= loss.item()

    # X = pred.detach().resize(1,943,1682)
    # optimizer.zero_grad()
    if(epoch%100==0):
        print("train loss: {}\t test loss: {} \t epoch:{}".format(loss,test_loss,epoch))
        x.append(epoch)
        train_loss__.append(loss)
        test_loss__.append(test_loss.detach().cpu())
        plt.plot(x,train_loss__,test_loss__,name="my_pytorch_model")