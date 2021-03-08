import numpy as np
import matplotlib.pyplot as plt
from awgn import awgn

#这里是假设A=1，H=1, B=0的情况
# 故动态模型 X(k) = X(k-1) + 噪声
#            Z(K) = X(k)
# 动态模型是一个常量

# intial parameters
'''n_iter = 50
sz = (n_iter,) # size of array
x = -0.37727 # truth value (typo in example at top of p. 13 calls this z)
z = np.random.normal(x,0.1,size=sz) # observations (normal about x, sigma=0.1)'''

t = np.linspace(0,1,2000)
x = np.piecewise(t,[t<1,t<0.8,t<0.3],
                    [lambda t : 1.5*np.sin(20*np.pi*t),
                     lambda t : np.sin(20*np.pi*t),
                    lambda t :np.sin(20*np.pi*t)+2*np.cos(13*np.pi*t)])
z = x + awgn(x,10)

Q = 0.00001 # process variance

# allocate space for arrays
xhat=np.zeros_like(t)      # a posteri estimate of x
P=np.zeros_like(t)         # a posteri error estimate
xhatminus=np.zeros_like(t) # a priori estimate of x
Pminus=np.zeros_like(t)    # a priori error estimate
K=np.zeros_like(t)         # gain or blending factor

R = 0.004 # estimate of measurement variance, change to see effect

# intial guesses
xhat[0] = 0.0
P[0] = 1.0

for k in range(1,len(t)):
    # time update
    xhatminus[k] = xhat[k-1]  #X(k|k-1) = AX(k-1|k-1) + BU(k) + W(k),A=1,BU(k) = 0
    Pminus[k] = P[k-1]+Q      #P(k|k-1) = AP(k-1|k-1)A' + Q(k) ,A=1

    # measurement update
    K[k] = Pminus[k]/( Pminus[k]+R ) #Kg(k)=P(k|k-1)H'/[HP(k|k-1)H' + R],H=1
    xhat[k] = xhatminus[k]+K[k]*(z[k]-xhatminus[k]) #X(k|k) = X(k|k-1) + Kg(k)[Z(k) - HX(k|k-1)], H=1
    P[k] = (1-K[k])*Pminus[k] #P(k|k) = (1 - Kg(k)H)P(k|k-1), H=1

plt.figure()
plt.subplot(311)
plt.plot(z, 'k-', label='noisy measurements')  # 测量值
plt.legend()
plt.subplot(312)
plt.plot(xhat, 'b-', label='a posteri estimate')  # 过滤后的值
plt.legend()
plt.subplot(313)
plt.plot(x,'r-',label = 'real value')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Voltage')
plt.show()