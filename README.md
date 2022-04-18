# my NMF vs sklearn NMF

## Shunyu Yao (Jason)

### 0. outline

​																1. compare RMSE loss between them(sklearn vs mine)

​																2. why sklearn is worse?

​																3. my algorithm

​																4. next step

### 1. RMSE loss on test set for both algorithm



#### 1.0 about RMSE

***R***oot ***M***ean ***S***qure ***E***rror

![查看源图像](https://ichi.pro/assets/images/max/724/1*9hQVcasuwx5ddq_s3MFCyw.gif)

#### 1.1 Sklearn

RMSE 2.23005 

convergent at Iterations about 1500

<img src="https://tva1.sinaimg.cn/large/e6c9d24egy1h1dogvbh55j20ss0lidgl.jpg" alt="image-20220417155155048" style="zoom:50%;" />

#### 1.2 My NMF model

RMSE 0.94012 

interation 300, still not convergent

<img src="https://tva1.sinaimg.cn/large/e6c9d24egy1h1dogy0escj20hs0dc3yt.jpg" alt="image-20220409203735335" width = 800 />

### 2. why sklearn lib is worse?

TOO faithful about the dataset

​	-**fit those 'zeros'**

​	cannot fully comprehend the question for this latent factor model

<img src="https://tva1.sinaimg.cn/large/e6c9d24egy1h1dogw83uvj20pa0m00ta.jpg" alt="Screenshot_2022-04-09-20-13-56-704_com.microsoft.office.onenote" style="zoom:50%;" />

### 3. My algorithm

#### 3.1 objective function

$\Sigma_j (RM_j - ERM_j)^2$

where j is a rating in training set, by user U on item I

I ignored those missing data, and focused on the existing entries

#### 3.2 parameter update

<img src="https://tva1.sinaimg.cn/large/e6c9d24egy1h1dogx6w55j214m0nswfk.jpg" alt="Screenshot_2022-04-10-11-37-24-274_com.microsoft.office.onenote" style="zoom:50%;" />

let objective function be F:

<img src="https://tva1.sinaimg.cn/large/e6c9d24egy1h1doh0tztsj210w0ql3zr.jpg" alt="Screenshot_2022-04-10-11-45-14-501_com.microsoft.office.onenote" style="zoom:50%;" />

### 4. Next step

##### 1. speed

Although sklearn's NMF is worth than my algorithm, it runs much faster than mine!!

* 10s for 20000 iterations
* 70s per iteration 

###### 1.1 multi-thread

​	GIL(global interpreter lock)  make sure that multi-thread wont mess up a variable

		- c, java?
		- Multi-process?

###### 1.2 for loop - too slow

​	go through each entry in training set  (SGD?)

##### 2. regularization

about overfitting

​	-no regularization is implemented for now （2 norm ?)

​	-but the loss on the test set is still diclining?

<img src="https://tva1.sinaimg.cn/large/e6c9d24egy1h1doh1vbiij21bd0u0409.jpg" alt="Screenshot_2022-04-09-20-31-25-861_com.microsoft.office.onenote" style="zoom:33%;" />

### 0. outline of ShunyuYao's presentation

1. Ignore zeros using library - Sklearn, Surprise, torchNMF
2. generate rating scores
3. GNN

### 1. Ignore zeros using library - Sklearn, Surprise, torchNMF

#### 1.1 sklearn

For support of missing data in sklearn, I have go through document and finally find something in sklearn's github repository.

In sklearn's official GitHub repository, issue #8447 and #8474 mentioned this approach, to ignore those missing data instead of fitting them in the process of decomposition.

<img src="https://tva1.sinaimg.cn/large/e6c9d24egy1h1dogglvhuj20t604oaak.jpg" alt="image-20220417150729155" width=400 />

<img src="https://tva1.sinaimg.cn/large/e6c9d24egy1h1dogirhzhj218e05c0tx.jpg" alt="image-20220417150755698" width=400 />

In 8447, they basically discussed the plausibility and validity for NMF in recommender system, some paper were presented. This issue is still open.

In 8474, after some discussion, some contributors decided to move this function to another library as an extra support library for sklearn. This issue is open and work-in-progress. [WIP]

#### 1.2 surprise

***SurPRISE***  stands for Simple Python RecommendatIon System Engine. It is basically build for this kind of stuff.

##### 1.2.1 RMSE loss

Trained with the same trianing sets, the same test sets and same amount of iterations, 300, and the same length of the W and H, 50, which suggest a movie has 50 attributes, such as the degree of horror, humorous, si-fi elements and etc.

<img src="https://tva1.sinaimg.cn/large/e6c9d24egy1h1dogkftmaj20we086gmk.jpg" alt="image-20220417151855013" style="zoom:50%;" />

Runs much faster than my model, probably using matrices multiplication instead of for loop. And the loss is lower than mine, 0.94, also. Since it has introduced regularization. 

##### 1.2.2 problem?

<img src="https://tva1.sinaimg.cn/large/e6c9d24egy1h1dogjmla8j20eq012a9v.jpg" alt="image-20220417151709911" style="zoom:50%;" />

Can only utilize one core of the CPU.

#### 1.3 torchNMF

Bases on torch.nn.module, can make full use of CPU, even CUDA devices.

But unfortunately, doesnot support the decomposition of matrices with missing data.

![image-20220417154044291](https://tva1.sinaimg.cn/large/e6c9d24egy1h1dogf5ryzj20k4028mx9.jpg)

#### 1.4 conclusion

|         Model         |      Accuracy       |                  Speed                   |
| :-------------------: | :-----------------: | :--------------------------------------: |
|        MyModel        |        ~0.94        |              Very very slow              |
|        Sklearn        | Cannot ignore zeros |                   Fast                   |
|       Surprise        |       0.9359        |     slow, can utilize only one core      |
|       TorchNMF        | Cannot ignore zeros |                Very fast                 |
| MyModel using PyTorch |  can ignore zeros   | very fast, can use CUDA for acceleration |
|                       |                     |                                          |

For next step I will try my model and algorithm using PyTorch , and then it will be accurate and fast at the same time!

### 2. generate rating score

<img src="https://tva1.sinaimg.cn/large/e6c9d24egy1h1doghscnlj21420c4ta5.jpg" alt="image-20220417161039560" style="zoom:50%;" />

### 3. with GNN?

In the session before last session, I mentioned that I will read more about GNN and work with RenZhu. 

I have just got a shallow understanding of GNN, the structure of each neuron, the idea behind it. And have tried some of them. Got the performence curve printed out. https://github.com/zhuty16/GES

For next step I will dive deeper in the field of GNN, get to know it's network structure and purpose of each part.










