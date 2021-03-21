import pywt
import matplotlib.pyplot as plt
import numpy as np
from awgn import awgn
from matplotlib import widgets  # 没用上的gui工具
import itertools  # 处理数组到列表的转换

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def peak_seek(data):  # 功能是找出fft变换后的几个峰值，也就是频率；
    pks = []
    for i in range(1, len(data) - 1):
        if abs(data[i]) > abs(data[i - 1]) and abs(data[i]) > abs(data[i + 1]) and abs(data[i]) > np.max(abs(data)) / 5:
            # 找出不是特别小（排除噪声）的极值点，就是需要保留的信号频率
            pks += [abs(data[i])]

    return pks


fs = 1000  # 采样频率
t = np.arange(0, 1, 1.0 / fs)
f_to = t / 1 * fs
N = fs * 1
# 信号
data = np.piecewise(t, [t < 1, t < 0.8, t < 0.3],
                    [lambda t: 1.5 * np.sin(200 * np.pi * t) + awgn(np.sin(200 * np.pi * t), 15),
                     lambda t: np.sin(200 * np.pi * t) + awgn(np.sin(200 * np.pi * t), 5),
                     lambda t: np.sin(200 * np.pi * t) + 2 * np.cos(130 * np.pi * t) + awgn(np.sin(200 * np.pi * t),
                                                                                            15)])
# 做两次让信号更加光滑
for i in range(2):

    wavename = "cgau8"
    totalscal = 256
    fc = pywt.central_frequency(wavename)  # 中心频率
    cparam = 2 * fc * totalscal
    scales = cparam / np.arange(totalscal, 1, -1)
    [coef, freqs] = pywt.cwt(data, scales, wavename, 1.0 / fs)
    fig, ax = plt.subplots()

    plt.contourf(t, freqs, abs(coef),cmap='hot',vmin=0,vmax=3.2)
    plt.title('（小尺度的cwt）鼠标左键选择，中键确认')
    xdata = plt.ginput(0, 0)  # 使用鼠标从图上获取分界点（其实人分不出来，这里有待改进）
    time = []
    for i in xdata:
        time += [i[0]]
    plt.show()

    '''
    #本想借助gui工具用户自己输入频谱的峰值，结果发现由于其特殊性（好吧因为我找不到为啥）
    函数执行时内部即使定义global变量也不能在函数外调用这个变量所以放弃了

    fig,ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)
    axbox = fig.add_axes([0.1, 0.05, 0.8, 0.075])


    def nums(expression):
        return expression

    text_box = widgets.TextBox(axbox,'input:')
    text_box.on_submit(nums)
    '''

    y = np.fft.fft(data)

    '''plt.subplot(412)
    plt.plot(f_to, abs(y))'''
    plt.title('origin in frequency domain')

    data_den = []

    skip = 0
    skip_next = int((time[0]) / np.max(t) * len(t))
    print(skip, skip_next)
    t_slit = t[skip:(skip_next + 1)]
    data_slit = data[skip:(skip_next + 1)]
    plt.subplot(511)
    plt.plot(t_slit, data_slit)
    f = t_slit / 1 * fs
    y = np.fft.fft(data_slit)
    plt.subplot(512)
    plt.plot(f, abs(y))
    print(abs(np.max(y)))
    peak = peak_seek(y)
    for i in range(len(y)):
        if abs(y[i]) not in peak:
            y[i] = 0

    data_slit_mod = np.fft.ifft(y)

    data_den += [data_slit_mod]

    for i in np.arange(len(time)):

        if i < len(time) - 1:
            skip = int(time[i] / np.max(t) * len(t))
            skip_next = int((time[i + 1]) / np.max(t) * len(t))
            print(skip, skip_next)
            t_slit = t[skip:(skip_next + 1)]
            data_slit = data[skip:(skip_next + 1)]
            plt.subplot(511)
            plt.plot(t_slit, data_slit)
            f = t_slit / 1 * fs
            y = np.fft.fft(data_slit)
            plt.subplot(512)
            plt.plot(f, abs(y))
            peak = peak_seek(y)

            for i in range(len(y)):
                if abs(y[i]) not in peak:
                    y[i] = 0

            data_slit_mod = np.fft.ifft(y)
            data_den += [data_slit_mod]



        elif i == len(time) - 1:
            skip = int(time[i] / np.max(t) * len(t))
            skip_next = len(t)
            print(skip, skip_next)
            t_slit = t[skip:(skip_next + 1)]
            data_slit = data[skip:(skip_next + 1)]
            plt.subplot(511)
            plt.plot(t_slit, data_slit)
            f = t_slit / 1 * fs
            y = np.fft.fft(data_slit)
            plt.subplot(512)
            plt.plot(f, abs(y))
            peak = peak_seek(y)
            for i in range(len(y)):
                if abs(y[i]) not in peak:
                    y[i] = 0

            data_slit_mod = np.fft.ifft(y)

            data_den += [data_slit_mod]

    plt.subplot(513)

    data_den_list = list(itertools.chain(*data_den))

    plt.plot(t, data_den_list[1:-1])
    plt.subplot(514)
    y = np.fft.fft(data_den_list)
    plt.plot(abs(y))
    data = data_den_list[1:-1]
    plt.subplot(515)
    fs = 1000  # 采样频率
    t = np.arange(0, 1, 1.0 / fs)
    data0 = np.piecewise(t, [t < 1, t < 0.8, t < 0.3],
                         [lambda t: 1.5 * np.sin(200 * np.pi * t),
                          lambda t: np.sin(200 * np.pi * t),
                          lambda t: np.sin(200 * np.pi * t) + 2 * np.cos(130 * np.pi * t)])
    plt.plot(data0)
    plt.title('clean data')
plt.show()






