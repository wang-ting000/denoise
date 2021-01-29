import numpy as np
import matplotlib.pyplot as plt
from awgn import awgn
import pywt


plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
from psd import psd


fs = 1024*8#采样频率
T0=0.125
N=T0*fs
t = np.arange(0,T0,1/fs)
data=np.sin(2*800*np.pi*t)+awgn(np.sin(2*800*np.pi*t),10)+1.5*np.sin(2*700*np.pi*t)

coeffs=pywt.wavedec(data,'db3',level=5)
'''plt.subplot(511)
plt.plot(coeffs[0])
plt.subplot(512)
plt.plot(coeffs[1])
plt.subplot(513)
plt.plot(coeffs[2])
plt.subplot(514)
plt.plot(coeffs[3])
plt.subplot(515)
plt.plot(coeffs[4])
plt.show()'''
coeffs[0]=0*coeffs[0]
coeffs[1]=0*coeffs[1]
#coeffs[2]=0*coeffs[2]
#coeffs[3]=0*coeffs[3]
#coeffs[4]=0*coeffs[4]
coeffs[5]=0*coeffs[5]
coeffs=[coeffs[0],coeffs[1],coeffs[2],coeffs[3],coeffs[4],coeffs[5]]
plt.subplot(411)
plt.plot(t,data)
plt.title('originnal data')
plt.subplot(412)
data1=pywt.waverec(coeffs,'db3')
plt.plot(t,data1)
plt.subplot(413)
psd(1/fs,T0,data)
plt.title('originnal data')
plt.subplot(414)
w=psd(1/fs,T0,data1)
plt.show()

data_ft=np.fft.fft(data1)
plt.plot(abs(data_ft))
plt.show()


threshold = np.max(abs(data_ft))
for i in range(len(data_ft)): ##通过FT将小于最大值1/10的频率值置为趋于0的数值
    if abs(data_ft[i]) <=threshold/9:   ##信噪比较小比如0的时候有随机性，有的时候由于噪声叠加可能导致误判
        data_ft[i]=10**(-40)
N = len(data_ft)
plt.plot(w[:int(N/2)],20*np.log10(abs(data_ft[:int(N/2)])))
plt.title('FT')
plt.show()
data_den = np.fft.ifft(data_ft)
plt.subplot( 411)
plt.plot(data)
plt.subplot(412)
plt.plot(data_den)
plt.subplot(413)
psd(1/fs,T0,data)
plt.subplot(414)
psd(1/fs,T0,data_den)
plt.show()






