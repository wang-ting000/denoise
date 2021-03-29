# -*- encoding: utf-8 -*-
"""
@File   : LMS.py
@Time   : 2021/3/26 21:19
@Author : Wang
"""
import numpy as np
import matplotlib.pyplot as plt
from awgn import awgn

def lms(u,dn,M,a,itr):
    '''

    :param u: 输入的加噪序列
    :param dn: 期望值（无噪）
    :param M: 滤波器阶数
    :param a: 步长因子
    :param itr: 迭代次数，也是信号的长度
    :return: 去噪后信号
    '''
    w = [[0]*M for i in range(itr)]##滤波器权重
    yn = np.ones_like(u)
    en = np.zeros_like(u)
    for k in range(M,itr-1):
        u_filter = u[k:(k-M):-1]##创造进入M阶滤波器的输入序列
        yn[k] = np.dot(w[k],u_filter)
        d = u_filter.mean()
        en[k] = d - yn[k]
        #en[k] = dn[k]-yn[k]
        #w[k+1] = w[k] + a*en[k]*u_filter
        w[k+1] = np.add(w[k],a*en[k]*u_filter)
    yn = np.ones_like(u)
    for k in range(M,itr):

        u_filter = u[k:(k - M):-1]  ##创造进入M阶滤波器的输入序列
        yn[k] = np.dot(w[k], u_filter)
    return yn,en

itr = 10240
t = np.linspace(0,2,itr)
x = np.sin(20*np.pi*t)*2 + 1.5*np.sin(33*np.pi*t)

noise = awgn(x,0)
input = x + noise
output,en = lms(input,x,64,0.00002,itr)
plt.subplot(311)
plt.plot(t,input)
plt.title('input')
plt.subplot(312)
plt.plot(t,output,label = 'denoised')
plt.plot(t,x,label = 'origin')
plt.title('output')
plt.legend()
plt.subplot(313)
plt.plot(t,en)
plt.title('error')
plt.show()

