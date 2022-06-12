import matplotlib.pyplot as plt

def plot(x,train_loss, test_loss,name="result.png"):
    plt.plot(x,train_loss,label="train_loss")
    plt.plot(x,test_loss,label="test_loss")
    plt.legend()
    plt.savefig(name)
    plt.show()