# -*- encoding: utf-8 -*-
"""
@File   : RLS.py
@Time   : 2021/3/30 14:53
@Author : Wang
"""
import numpy as np
import matplotlib.pyplot as plt
from awgn import awgn

def rls(lmd,M,u,d,delta,itr):
    '''

    :param lmd: 遗忘因子
    :param M: 滤波器阶数
    :param u: 输入信号
    :param d: 期望信号
    :param delta: 初始化参数P(0)
    :param itr:信号的个数
    :return:去噪的信号
    '''
    w =[[0]*M for i in range(itr)]
    I = np.eye(M)
    p = I/delta
    for i in range(M,itr):#迭代itr次
        x = u[i:i-M:-1]
        alpha = d[i] -np.dot(x,w[i-1])
        g = np.dot(p,x)/(lmd+np.dot(np.dot(x,p),x))
        p = 1/lmd*p-np.multiply(np.multiply(g,x),p)*1/lmd
        w[i] = w[i-1] + alpha*g
    yn = np.zeros(itr)
    #return w[itr-2],alpha,x
    for k in range(M, itr):
        u_filter = u[k:(k - M):-1]  ##创造进入M阶滤波器的输入序列
        yn[k] = np.dot(w[k], u_filter)
    return yn

itr = 1000
M = 64
lmd = 1
delta = 1e-7
t = np.linspace(0,1,itr)
#d = np.sin(20*np.pi*t)+1.5*np.sin(33*np.pi*t)
d = np.piecewise(t,[t<1,t<0.8,t<0.3],
                    [lambda t : 1.5*np.sin(100*np.pi*t),
                     lambda t : np.sin(200*np.pi*t),
                    lambda t :np.sin(59*np.pi*t)+2*np.cos(33*np.pi*t)])
u = d+awgn(d,0)
output = rls(lmd,M,u,d,delta,itr)
plt.subplot(211)
plt.plot(t,u,label = 'noised')
plt.subplot(212)
plt.plot(t,output,label = 'output')
plt.plot(t,d,label = 'origin')
plt.legend()
plt.show()
