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



