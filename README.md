# denoise 

<img align="right" img width = '400' height = '200' src="https://i.loli.net/2021/04/05/I9VNQMWDfPZb6iL.png"/>

去噪的方法很多，其中小波去噪、自适应去噪、卡尔曼滤波器都有不同的特点。白噪声因为其频谱范围宽，传统的滤波器无法达到较好的滤波目的。
## ![](https://img.shields.io/badge/1-%E5%B0%8F%E6%B3%A2%E5%8F%98%E6%8D%A2-yellowgreen)
  * 小波去噪的经典方法有：阈值法，小波分解法，小波包分解法

  * 连续小波变换得到的系数是时间和频率，根据小波系数可以画出时频图如右图：
  <img align="right" img width = '400' height = '200' src="https://i.loli.net/2021/04/05/kXFeMiQhnpfr4A9.png"/>   

### ![](https://img.shields.io/badge/1.1-%E9%98%88%E5%80%BC%E6%B3%95%E5%8E%BB%E5%99%AA-blue)


* 原理：经过正交小波变换小波变换后的小波系数中，能量集中在少数小波系数中，也即幅值比较大的小波系数绝大多数是有用信号，而幅值较小的一般是噪声；寻找到一个合适的阈值，将小于阈值的小波系数进行相应处理，然后根据处理后的小波系数还原出有用信号

* 步骤：
    * 正交小波变换，选取一个正交小波和分解层数N
    * 对测量信号的每一层高频系数通过阈值函数处理，低频系数不处理(低频代表着概貌)
    * 对处理后的小波系数进行重构
    
* 软阈值函数：会丢掉信号的一些特征  
![](https://www.researchgate.net/profile/Wenhui-Wang-10/publication/245331181/figure/fig4/AS:667593513922565@1536178105298/Thresholding-schemes-a-hard-thresholding-and-b-soft-thresholding.png)

* 硬阈值函数：会产生突变

* 效果：![](https://i.loli.net/2021/04/05/loBeOPIRGUSZm8f.png)

### ![](https://img.shields.io/badge/1.2-%E5%B0%8F%E6%B3%A2%E5%88%86%E8%A7%A3-blue)

* 原理：多尺度小波分解相当于一个高通滤波器和若干个带通滤波器，将每一级的低频系数继续分解；如果将噪声对应的系数清除，或者将信号对应的系数挑出来进行重构，就可以得到去噪后的信号

* 效果:![](https://i.loli.net/2021/04/05/LNrkPZHvYzxD5uB.png)

* ps:综合考虑小波分解的分辨率以及信号频率等因素，选取合适的分解层数和采样率可以更好地还原信号，这个可以通过遍历法选取相关系数最大的一次  
* 未解决的问题：会出现一些谐波，怀疑是令某些小波系数为0产生的突变点

### ![](https://img.shields.io/badge/1.3-%E5%B0%8F%E6%B3%A2%E5%88%86%E8%A7%A3%E9%87%8D%E6%9E%84%E4%B8%8E%E5%82%85%E9%87%8C%E5%8F%B6%E5%8F%98%E6%8D%A2%E5%8E%BB%E5%99%AA%E7%BB%93%E5%90%88-blue)

* 基本思路：先在通过DWT的时频图中判断出频率分段点，然后结合matplotlib的GUI功能获取其坐标，进而分时间进行FT滤波，最后拼接起来

* 不足：在分界点处的信号突变点会出现相位错误

* 效果:
* ![](https://i.loli.net/2021/04/05/pF3TtWxjSwqaBYo.png)


## ![](https://img.shields.io/badge/2-%E8%87%AA%E9%80%82%E5%BA%94%E6%BB%A4%E6%B3%A2-yellowgreen)
* 主要包含基于维纳滤波器的 `LMS算法` 、 `递推最小二乘的RLS算法` 和 `卡尔曼滤波器 `  

* [文档](小波变换的延申.pdf)
### ![](https://img.shields.io/badge/3-%E7%BB%B4%E7%BA%B3%E6%BB%A4%E6%B3%A2%E5%99%A8-yellowgreen)  

原理是使估计值和期望值之间的均方误差最小化

* `前提假设：信号和噪声都是平稳的`

* 维纳滤波器可用于解决非因果滤波器`需要无限量过去和未来的数据`、因果滤波器`需要使用无限量过去的数据`和FIR的问题，但是维纳滤波器一般用于解决第二种问题  
* 对于因果问题，维纳滤波器的解决方式：
    * <img src="http://latex.codecogs.com/svg.latex?\overrightarrow{h}_{opt}=\overrightarrow{R}_{xx}^{-1}\overrightarrow{R}_{xs}" />  

    * <img src="http://latex.codecogs.com/svg.latex?\overrightarrow{R}_{xs}=\begin{bmatrix} \overrightarrow{R}_{xs}(0) & \overrightarrow{R}_{xs}(1) ... & \overrightarrow{R}_{xs}(M) \end{bmatrix}^T" />  

    * <img src="http://latex.codecogs.com/svg.latex?\overrightarrow{R}_{xx}=\begin{bmatrix} \overrightarrow{R}_{xx}(0) & \overrightarrow{R}_{xx}(1) & ... & \overrightarrow{R}_{xx}(M) \\ \overrightarrow{R}_{xx}(1) & \overrightarrow{R}_{xx}(2) & ... & \overrightarrow{R}_{xx}(M-1) \\ ... & ... &... & ... \\ \overrightarrow{R}_{xx}(M) & \overrightarrow{R}_{xx}(M-1) & ... & \overrightarrow{R}_{xx}(0)\end{bmatrix}^T" />  

* 根据这个公式求出最佳的h(t)，含噪声信号和它的卷积就是期望信号  

* 最后的效果如下：由于对称矩阵的特点，以至于有一半的信号未被恢复，对于平稳信号而言，并未造成信号的丢失（个人认为是的吧） 

* ![wiener](https://i.loli.net/2021/04/03/DVKAtp64517gNod.png)

### ![](https://img.shields.io/badge/4-%E5%9F%BA%E4%BA%8E%E7%BB%B4%E7%BA%B3%E6%BB%A4%E6%B3%A2%E5%99%A8%E7%9A%84LMS%E7%AE%97%E6%B3%95-yellowgreen)

* 一种最陡下降算法的改进算法， 是在维纳滤波理论上运用速下降法后的优化延伸，最终收敛到维纳滤波器  

* <img align="right" img width = '500' height = '300' src="https://upload.wikimedia.org/wikipedia/commons/6/62/Lms_filter.svg"/>

* 优点：在自适应均衡的时候就可以很快的跟踪到信道的参数，减少了训练序列的发送时间，从而提高了信道的利用率；易于操作，性能稳健  

* 缺点：收敛速度慢  
  

* 算法实现： 
* <img src="http://latex.codecogs.com/svg.latex?e(n)=d(n)-y(n); " />  
* <img src="http://latex.codecogs.com/svg.latex?y(n)=\overrightarrow{x}^T(n)\overrightarrow{w}^T(n)"/>
* <img src="http://latex.codecogs.com/svg.latex?y(n)=\overrightarrow{w}(n+1)=\overrightarrow{w}(n)+2*\mu*e(n)*\overrightarrow{x}(n)"/>
* 收敛效率：
    * 如果步长μ的取值较大，则收敛速度快，因此μ需要满足一定的范围：0<μ<2/λ<sub>max</sub>,其中λ<sub>max</sub>是自相关矩阵的最大特征值；如果未达到这个条件，滤波器将无法收敛，系统也是非稳定的



### ![](https://img.shields.io/badge/5-%E5%8D%A1%E5%B0%94%E6%9B%BC%E6%BB%A4%E6%B3%A2%E5%99%A8-yellowgreen)  
  <img align="right" img width = '300' height = '200' src="https://upload.wikimedia.org/wikipedia/commons/b/b6/Kalman_filter_model.png"/>  

* 又称为线性二次滤波LQE，使用随时间观察到的一系列测试值，并产生未知变量的估计值  


* 应用领域： **信号处理**，**机器人运动预测**   


![](https://img.shields.io/badge/-%E7%BA%AF%E6%97%B6%E5%9F%9F%E6%BB%A4%E6%B3%A2%E5%99%A8-lightgrey)  

![](https://img.shields.io/badge/-%E9%94%81%E7%9B%B8%E7%8E%AF%E5%B0%B1%E6%98%AF%E4%B8%80%E4%B8%AAKarman%E6%BB%A4%E6%B3%A2%E5%99%A8%EF%BC%81-red)

* karman滤波器的状态由两个变量表示：
* <img src="http://latex.codecogs.com/svg.latex?\hat{\textbf{x}}_{k|k}" /> **在时刻k的状态估计**
* <img src="http://latex.codecogs.com/svg.latex?\textbf{P}_{k|k}" /> **后验估计误差协方差矩阵，度量估计值的精确程度**
* **预测**
    * <img src="http://latex.codecogs.com/svg.latex?\hat{\textbf{x}}_{k|k-1}=\textbf{F}_k\hat{\textbf{x}}_{k-1|k-1}+\textbf{B}_k\textbf{U}_k" /> (**预测状态**)
    * <img src="http://latex.codecogs.com/svg.latex?\textbf{P}_{k|k-1}=\textbf{F}_k\textbf{P}_{k-1|k-1}\textbf{F}_k^T+\textbf{Q}_k" />  (**预测估计协方差矩阵**)
* **更新**  

    `先算出：`
    * <img src="http://latex.codecogs.com/svg.latex?\tilde{\textbf{y}}_k=\textbf{z}_k-\textbf{H}_k\hat{\textbf{x}}_{k|k-1}" /> (**测量残差**)
    * <img src="http://latex.codecogs.com/svg.latex?\textbf{S}_k=\textbf{H}_k\textbf{P}_{k|k-1}\textbf{H}_k^T+\textbf{R}_k" /> (**测量残差协方差**)
    * <img src="http://latex.codecogs.com/svg.latex?\textbf{K}_k=\textbf{P}_{k|k-1}\textbf{H}_k^T\textbf{S}_k^{-1}" /> (**最佳卡尔曼增益**)  
    
    `然后用它们来更新滤波器变量x,p`
    * <img src="http://latex.codecogs.com/svg.latex?\hat{\textbf{x}}_{k|k}=\hat{\textbf{x}}_{k-1|k-1}+\textbf{K}_k\widetilde{\textbf{y}}_k" /> (**更新的状态估计**)
    * <img src="http://latex.codecogs.com/svg.latex?\textbf{P}_{k|k}=(I-\textbf{K}_k\textbf{H}_k)\textbf{P}_{k|k-1}" />  (**预测估计协方差矩阵**)

### ![](https://img.shields.io/badge/6-%E5%9F%BA%E4%BA%8E%E5%B8%8C%E5%B0%94%E4%BC%AF%E7%89%B9--%E9%BB%84%E5%8F%98%E6%8D%A2%E7%9A%84%E6%A8%A1%E6%80%81%E5%88%86%E8%A7%A3-yellowgreen)
* 将数据分解为本质模态函数(`IMF`),这个过程称为经验模态分解(`EMD`),然后对IMF进行希尔伯特变换以得到数据的频率
* ![](https://img.shields.io/badge/-%E4%B8%8D%E5%90%8C%E4%BA%8EFT%2C%E6%98%AF%E4%B8%80%E7%A7%8D%E7%AE%97%E6%B3%95%E8%80%8C%E9%9D%9E%E7%90%86%E8%AE%BA%E5%B7%A5%E5%85%B7-brightgreen)
* **IMF**:一个函数若属于IMF，代表其波形局部对称于零平均值
* **EMD**:将信号分解成IMF，具体过程见[👉](https://zh.wikipedia.org/zh-cn/%E5%B8%8C%E7%88%BE%E4%BC%AF%E7%89%B9-%E9%BB%83%E8%BD%89%E6%8F%9B)


### ![](https://img.shields.io/badge/7-LS--filter-yellowgreen)
* RLS算法和LMS算法都属于LS滤波器的衍生  
* LMS算法和模型无关，跟踪特性好；RLS算法和模型有关，收敛速度快、精度高但同时计算时间也长
* LMS是一种特殊情况下的RLS
* `初始化`：
* <img src="http://latex.codecogs.com/svg.latex?P(0)=\delta*I" />
* <img src="http://latex.codecogs.com/svg.latex?\hat{W}(0)=0, \lambda = 1" />
* `迭代`:
* <img src="http://latex.codecogs.com/svg.latex?k(n)=\frac{\lambda^{-1}P(n-1)u(n)}{1+\lambda^{-1}u^H(n)P(n-1)u(n)}" />
* <img src="http://latex.codecogs.com/svg.latex?\xi(n)=d(n)-\hat{W}^H(n-1)u(n)" />
* <img src="http://latex.codecogs.com/svg.latex?P(n)=\lambda^{-1}(P(n-1)-k(n)u^H(n)P(n-1)" />



**总的来说：**   

- [x] FFT去噪适用于平稳信号，方法容易，在简单情况下处理效果不错，但是由于余弦函数的特点，不能够较好地拟合出发生剧烈变化的位置，会出现吉布斯现象  

- [x] WT则可以很好地解决这一问题，因为它的基是各类小波，可以在时间和频率尺度上有不同的分辨率，但是依旧受到[海森堡不确定性原理](https://www.youtube.com/watch?v=TDfag7uYLww)的制约，所以时频分辨率不能同时达到最大。小波变换适于分析非平稳信号，此时用功率谱密度呈现信号特点已失去了意义，数学角度上也很好解释：非平稳信号不再有自相关函数只和时间差τ有关系，所以也就无法求得自相关函数的FT了


- [x] HHT使用EMD分解的方法，不受到海森堡不确定性的制约,因此可以同时在时频维度达到较好的分辨率，相比于小波变换对频率的定位更加精准了。此外，借助Hilbert变换求得相位函数，再对相位函数求导产生瞬时频率，这样求出的瞬时频率是局部性的，而傅立叶变换的频率是全局性的，小波变换的频率是区域性的。

- [x] 自适应滤波器可以根据先验信息对信号进行处理进而获得符合期望值的数据，这一特点使之可以在未知的环境中工作。主要包含三种算法：
    - [x] **维纳滤波器**，适用于平稳信号，利用自相关函数和互相关函数得到最优维纳解；
    - [x] **LMS算法**，采用迭代的方式，最终收敛到维纳滤波器，LMS算法只是用`以前各时刻的抽头参量`等作该时刻数据块估计时的平方误差均方最小的准则，而未用现时刻的抽头参量等来对以往各时刻的数据块作重新估计后的累计平方误差最小的准则，所以LMS算法对非平稳信号的适应性差；
    - [x] **卡尔曼滤波器**，不同于维纳滤波器假设所有信号受到的噪声是一样的，卡尔曼滤波器利用上一个时刻信号状态和当前状态来更新滤波器参数，因此适用于非平稳信号，一般用在预测模型中；
    - [x] **RLS算法**，基本思想是力图使在每个时刻对所有已输入信号而言重估的平方误差的加权和最小，这使得RLS算法对非平稳信号的适应性要好。与LMS算法相比，RLS算法采用时间平均，因此，所得出的最优滤波器依赖于用于计算平均值的样本数


