import numpy as np

def corrMat(array1,array2):
    '''

    :param array1: array1
    :param array2: array2
    :return: correlate matrix
    '''

    def crossCorr(array1, array2, l):
        '''

        :param array1: 数组1
        :param array2: 数组2
        :param l: 移动的距离
        :return: 移动l的相关系数
        plus::此函数应用于两个数组size相等的情景
        '''
        array1 = np.array(array1)
        array2 = np.array(array2)
        M = len(array1)
        S_12 = 0
        for i in range(0, M - l ):
            S_12 += array1[i]*array2[i + l]
        return S_12

    def crossCorrlist(array1, array2):
        '''

        :param array1: array1
        :param array2: array2
        :return: a row of correlate matrix
        '''
        list = []
        for i in range(len(array1)):
            list.append(crossCorr(array1, array2, i))  # 相关矩阵的一行
        return list

    N = len(array1)
    R = list(np.zeros_like(array1))
    for i in range(len(array1)):  # 每一行
        if i == 0:
            R[i] = crossCorrlist(array1, array2)
        else:
            R[i] = [x for x in
                    np.concatenate((crossCorrlist(array1, array2)[i:0:-1],
                                    crossCorrlist(array1, array2)[0:(N - i)]))]

    print(np.shape(R))
    return np.matrix(R)

def corrList(array1,array2):
    '''

    :param array1: array1
    :param array2: array2
    :return: correlate list
    '''

    def crossCorr(array1, array2, l):
        '''

        :param array1: 数组1
        :param array2: 数组2
        :param l: 移动的距离
        :return: 移动l的相关系数
        plus::此函数应用于两个数组size相等的情景
        '''
        array1 = np.array(array1)
        array2 = np.array(array2)
        M = len(array1)
        S_12 = 0
        for i in range(0, M - l ):
            S_12 += array1[i]*array2[i + l]
        return S_12


    list = []
    for i in range(len(array1)):
        list.append(crossCorr(array1, array2, i))  # 相关矩阵的一行
    return np.matrix(list).T #转置

'''R_xx = corrMat([1,2,3],[1,2,3])
R_xs = corrList([1,2,3],[2,3,4])
h = np.linalg.pinv(R_xx)*R_xs
print(np.array(h))
result = np.convolve(np.array(h).flatten(),[1,2,3],'full')
print(result)'''
