'''
画cwt的小波系数的时频图
'''

import pywt
import matplotlib.pyplot as plt
import numpy as np
from awgn import awgn

from mpl_toolkits.mplot3d import Axes3D


fs = 1000#采样频率
t = np.arange(0,1,1.0/fs)
data = np.piecewise(t,[t<1,t<0.8,t<0.3],
                    [lambda t : np.sin(200*np.pi*t),
                     lambda t : 2*np.sin(500*np.pi*t),
                     lambda t :np.sin(200*np.pi*t)])
wavename = "cgau8"
totalscal = 256
fc = pywt.central_frequency(wavename)#中心频率
print(fc)
cparam = 2 * fc * totalscal
scales = cparam/np.arange(totalscal,1,-1)
[coef, freqs] = pywt.cwt(data,scales,wavename,1.0/fs)#连续小波变换
plt.subplot(311)
plt.plot(t, data)
plt.xlabel("time(s)")
plt.title("Time spectrum")
plt.subplot(312)
plt.contourf(t, freqs, abs(coef),cmap='hot',vmin=0,vmax=3.2)
plt.ylabel("freq(Hz)")
plt.xlabel("time(s)")
plt.colorbar()
plt.subplot(313)
data = np.piecewise(t,[t<1,t<0.8,t<0.3],
                    [lambda t : np.sin(200*np.pi*t),
                     lambda t : 2*np.sin(500*np.pi*t),
                     lambda t :np.sin(200*np.pi*t)+awgn(np.sin(200*np.pi*t),-10)])
[coef, freqs] = pywt.cwt(data,scales,wavename,1.0/fs)#连续小波变换
plt.contourf(t, freqs, abs(coef),cmap='hot',vmin=0,vmax=3.2)
plt.ylabel("freq(Hz)")
plt.xlabel("time(s)")
plt.title('noise added')
plt.colorbar()
plt.show()
