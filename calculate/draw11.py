import csv  # 导入csv模块
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

file_name1 = r'C:\Users\yuancao\Desktop\dataprepare _copy\B06all\B06_copy.csv'
unrate1 = pd.read_csv(file_name1)  # 使用pandas读取数据
unrate1['date'] = pd.to_datetime(unrate1['date'])


setting = r'C:\Users\yuancao\Desktop\dataprepare _copy\B06all\prediction_log_copy.csv'
unrate2 = pd.read_csv(setting)  # 使用pandas读取数据
# unrate['date'] = pd.to_datetime(unrate['date'])

# unrate = pd.read_csv('TP.csv')  # 使用pandas读取数据
# unrate['date'] = pd.to_datetime(unrate['date'])

# reals = np.load(setting+'/real_prediction_JFNG_data_15min_2021.npy')


path_x = unrate1[:]
path_y = unrate2[:]

plt.figure()
plt.figure(figsize=(10, 7))

plt.plot(path_x['tp'])
plt.plot(path_y['tp'])

# plt.plot(path_x['date'], path_x['tp'])

# plt.plot(reals[0,:,-1], label='Prediction')

# plt.xticks(rotation=30)
#print help(plt.xticks)
plt.show()



