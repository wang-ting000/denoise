import  numpy as np
import matplotlib.pyplot as plt
from awgn import awgn
import random
from corrMatrix import corrMat,corrList



fs = 500 # 采样频率
T0 = 1
N = T0 * fs
t = np.arange(0, T0, 1 / fs)
signal = np.sin(40 * np.pi * t)
data = signal + awgn(signal,10)
plt.show()

plt.subplot(211)
plt.plot(t,data)
plt.title('original')



'''
##直接使用维纳滤波函数
plt.subplot(212)
from scipy import signal
result = signal.wiener(data)
result = signal.wiener(result)
result = signal.wiener(result)
plt.plot(t,result)
plt.title('denoised')
plt.show()
'''



#
'''R_xs = np.correlate(data,signal,'full')
R_xx = np.correlate(data,data,'full')
#h = R_xs/R_xx
h = np.dot(1/R_xx,R_xs)'''



R_xx = corrMat(data,data)
R_xs = corrList(data,signal)
h = np.linalg.pinv(R_xx)*R_xs

print(np.shape(h))
result = np.convolve(np.array(h).flatten(),data,'same')

plt.subplot(212)
plt.plot(t,result)
plt.title('denoised')
plt.show()

