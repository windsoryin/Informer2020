"""
方法一：自带的库fastdtw
"""
from fastdtw import fastdtw
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.spatial.distance import euclidean

file_name1 = r'C:\Users\yuancao\Desktop\dataprepare\5_30\B04_end.csv'
file_name2 = r'C:\Users\yuancao\Desktop\dataprepare\5_30\prediction_log_end.csv'

data1 = np.genfromtxt(file_name1, delimiter=',')
data2 = np.genfromtxt(file_name2, delimiter=',')

column1_data1 = data1[1:, 6]
column1_data2 = data2[1:, 1]

x = np.array(column1_data1)
y = np.array(column1_data2)

x = x.reshape(-1, 2)
y = y.reshape(-1, 2)

print(column1_data1)
print()
print(column1_data2)

setting1 = r'C:\Users\yuancao\Desktop\database\informer_JFNG_data_15min_ftMS_sl48_ll24_pl12_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0\true.npy'
setting2 = r'C:\Users\yuancao\Desktop\database\informer_JFNG_data_15min_ftMS_sl48_ll24_pl12_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0\pred.npy'

column_data1 = np.load(setting1)
column_data2 = np.load(setting2)
column_data1 = column_data1.reshape(-1, 2)
column_data2 = column_data2.reshape(-1, 2)

tdi = 0
N_output = column_data2.shape[0]
print(N_output)
distance, path = fastdtw(column_data1, column_data2, dist=euclidean)
Dist = 0
for i, j in path:
    Dist += (i - j) * (i - j)
tdi += Dist / (N_output * N_output)

print(distance)
# print(path)
print(tdi)

"""
方法二
"""
# import numpy as np
#
#
# def dtw(s, t):
#     """
#     :param s: 第一个序列
#     :param t: 第二个序列
#     :return:
#     """
#     n, m = len(s), len(t)
#
#     # 构建n行m列矩阵
#     dtw_matrix = np.zeros((n + 1, m + 1))
#     dtw_matrix.fill(np.inf)
#     dtw_matrix[0, 0] = 0  # 路径左上-->右下
#
#     # dtw_matirx[i, j] is the distance between s[1:i] and t[1:j] with the best alignment.
#
#     for i in range(1, n + 1):
#         for j in range(1, m + 1):
#             cost = np.abs(s[i - 1] - t[j - 1])  # 计算绝对距离
#             # 找到当前索引方块的top,left,top_left方向的累积损失的最小值
#             last_min = np.min([dtw_matrix[i, j - 1], dtw_matrix[i - 1, j], dtw_matrix[i - 1, j - 1]])
#             dtw_matrix[i, j] = cost + last_min
#     return dtw_matrix
#
#
# def DTWDistance(s1, s2):
#     DTW = {}
#
#     for i in range(len(s1)):
#         DTW[(i, -1)] = float('inf')
#     for i in range(len(s2)):
#         DTW[(-1, i)] = float('inf')
#     DTW[(-1, -1)] = 0
#
#     for i in range(len(s1)):
#         for j in range(len(s2)):
#             dist = (s1[i] - s2[j]) ** 2
#             DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])
#
#     return np.sqrt(DTW[len(s1) - 1, len(s2) - 1])
#
#
# file_name1 = r'C:\Users\yuancao\Desktop\dataprepare\5_30\B04_end.csv'
# file_name2 = r'C:\Users\yuancao\Desktop\dataprepare\5_30\prediction_log_end.csv'
# file_name3 = r'C:\Users\yuancao\Desktop\dataprepare _copy4\5_29_copy\B05\B05_end.csv'
# file_name4 = r'C:\Users\yuancao\Desktop\dataprepare _copy4\5_29_copy\B05\prediction_log_end.csv'
# file_name5 = r'C:\Users\yuancao\Desktop\dataprepare _copy4\5_29_copy\B06\B06_end.csv'
# file_name6 = r'C:\Users\yuancao\Desktop\dataprepare _copy4\5_29_copy\B06\prediction_log_end.csv'
# file_name7 = r'C:\Users\yuancao\Desktop\dataprepare _copy4\5_29_copy\B07\B07.csv'
# file_name8 = r'C:\Users\yuancao\Desktop\dataprepare _copy4\5_29_copy\B07\prediction_log.csv'
# file_name9 = r'C:\Users\yuancao\Desktop\dataprepare _copy4\5_29_copy\B08\B08.csv'
# file_name10 = r'C:\Users\yuancao\Desktop\dataprepare _copy4\5_29_copy\B08\prediction_log.csv'
#
#
# data1 = np.genfromtxt(file_name1, delimiter=',')
# data2 = np.genfromtxt(file_name2, delimiter=',')
# data3 = np.genfromtxt(file_name3, delimiter=',')
# data4 = np.genfromtxt(file_name4, delimiter=',')
# data5 = np.genfromtxt(file_name5, delimiter=',')
# data6 = np.genfromtxt(file_name6, delimiter=',')
# data7 = np.genfromtxt(file_name7, delimiter=',')
# data8 = np.genfromtxt(file_name8, delimiter=',')
# data9 = np.genfromtxt(file_name9, delimiter=',')
# data10 = np.genfromtxt(file_name10, delimiter=',')
#
# column1_data1 = data1[1:, 6]
# column1_data2 = data2[1:, 1]
# column1_data3 = data3[1:, 6]
# column1_data4 = data4[1:, 1]
# column1_data5 = data5[1:, 6]
# column1_data6 = data6[1:, 1]
# column1_data7 = data7[1:, 6]
# column1_data8 = data8[1:, 1]
# column1_data9 = data9[1:, 6]
# column1_data10 = data10[1:, 1]
#
# # print(column1_data1)
# # print(column1_data2)
#
# dtw1 = dtw(column1_data1, column1_data2)
# print("B04_dtw的矩阵如下：")
# print(dtw1)
#
# print("---------------------------------------------")
# dtw2 = dtw(column1_data3, column1_data4)
# print("B05_dtw的矩阵如下：")
# print(dtw2)
#
# print("---------------------------------------------")
# dtw3 = dtw(column1_data5, column1_data6)
# print("B06_dtw的矩阵如下：")
# print(dtw3)
#
# print("---------------------------------------------")
# dtw4 = dtw(column1_data7, column1_data8)
# print("B07_dtw的矩阵如下：")
# print(dtw4)
#
# print("---------------------------------------------")
# dtw5 = dtw(column1_data9, column1_data10)
# print("B08_dtw的矩阵如下：")
# print(dtw5)
#
# print("---------------------------------------------")
#
# dtw_dist_12 = DTWDistance(column1_data1, column1_data2)
# print('dtw_dist_B04 = {0:6.2f}'.format(dtw_dist_12))
# dtw_dist_34 = DTWDistance(column1_data3, column1_data4)
# print('dtw_dist_B05 = {0:6.2f}'.format(dtw_dist_34))
# dtw_dist_56 = DTWDistance(column1_data5, column1_data6)
# print('dtw_dist_B06 = {0:6.2f}'.format(dtw_dist_56))
# dtw_dist_78 = DTWDistance(column1_data7, column1_data8)
# print('dtw_dist_B07 = {0:6.2f}'.format(dtw_dist_78))
# dtw_dist_910 = DTWDistance(column1_data9, column1_data10)
# print('dtw_dist_B08 = {0:6.2f}'.format(dtw_dist_910))




# import numpy as np
# from scipy.spatial.distance import euclidean
#
# from fastdtw import fastdtw
#
# x = np.array([[1,1], [2,2], [3,3], [4,4], [5,5]])
# y = np.array([[2,2], [3,3], [4,4]])
# distance, path = fastdtw(x, y, dist=euclidean)
# print(distance)
