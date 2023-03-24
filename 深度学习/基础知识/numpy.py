import numpy as np
#常规生成numpy矩阵的方法
array = np.array([
    [1, 3, 5],[4, 6, 9]
])
print(array)
#numpy矩阵有三个参数：维数、行列、元素个数
print('number of dim:', array.ndim)#维数
print('shape:', array.shape)#行，列
print('size:', array.size)#元素个数

#生成3*2矩阵
array0 = np.array([
    [0, 2], [4, 7], [8, 10]
])
print(array0)

#生成矩阵的同时，能同时设定矩阵数据类型：有int32等
array1 = np.array([2, 9, 4], dtype = np.int32)
print(array1)
print(array1.dtype)

#生成a*b的零矩阵
array2 = np.zeros((4,4),dtype = np.int32)
print(array2)

#生成a*b的数据为1的矩阵
array3 = np.ones((4, 4), dtype = np.int32)
print(array3)

#生成a*b的空矩阵
array4 = np.empty((4,4))
print(array4)

#arrange(a,b,c),生成a-（b-1）的线性矩阵，步长为c
array5 = np.arange(10,21,2)
print(array5)

#reshape：线性矩阵reshape为a*b的矩阵
array6 = array3.reshape(2,8)
print(array6)

#linspace：将a-b分成20个数
array7 = np.linspace(0,19,20)
print(array7)

array8 = array7.reshape(4,5)
print(array8)

#矩阵合并
print(array2,"\n", array3)

#减法：每个元素对应相减
array9 = array2 - array3
print(array9)

#矩阵乘法
array10 = array.dot(array0)
print("乘法",array10)

#矩阵里每个元素平方
array17 = array10 ** 2
print(array17)

#矩阵里每个元素取sin值
array18 = np.sin(array17)
print(array18)

#输出每个元素的判断结果
print(array18 < 0)

#输出每个元素的判断结果
array20 = array10 * array10
print(array20 == array17)

#矩阵乘法的另一种表现形式
array24 = np.dot(array, array0)
print(array24)

#.random.random(a,b)取得随机矩阵
array25 = np.random.random((2,4))
#array求和
print(np.sum(array25, axis = 0))#axis = 1,对行操作；axis = 0，对列操作
#找最小元素值
print(np.min(array25))
#找最大元素值
print(np.max(array25))
print("array25 = ", array25)

#矩阵转置
array26 = array25.T
array26 = array25.transpose()
array26 = np.transpose(array25)
print(array26)

ed = np.array([12])
print(ed.ndim)
