import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 读取数据
setting = r'C:\Users\yuancao\Desktop\database\informer_JFNG_data_15min_ftMS_sl48_ll24_pl12_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0'
trues = np.load(setting+'/true.npy',allow_pickle= True)
preds = np.load(setting+'/pred.npy', allow_pickle=True)

true_length = len(trues)
pred_length = len(preds)
print(true_length)
print(pred_length)

"""
第一种情况：正确预报
在12个点内，真实值有一个或多个个大于0.5的值 && 预测值有一个或多个大于0.5的值（降雨）
在12个点内，真实值和预测值的值都小于0.5（不降雨）
此时，计数器加1
"""
counter1 = 0
for i, j in zip(range(trues.shape[0]), range(preds.shape[0])):
    tmp1 = trues[31110, :, :]
    tmp2 = preds[31110, :, :]
    if any(itr1 > 0.5 for itr1 in tmp1) and any(itr2 > 0.5 for itr2 in tmp2):
        counter1 = counter1 + 1
        # print(i)

    # plt.figure()
    # plt.figure(figsize=(15, 10))
    # plt.plot(tmp1[:].reshape(-1), label='GroundTruth')
    # plt.plot(tmp2[:].reshape(-1), label='Prediction')
    # plt.legend()
    # plt.show()

counter4 = 0
for i, j in zip(range(trues.shape[0]), range(preds.shape[0])):
    tmp1 = trues[15000, :, :]
    tmp2 = preds[15000, :, :]
    if all(itr1 < 0.5 for itr1 in tmp1) and all(itr2 < 0.5 for itr2 in tmp2):
        counter4 = counter4 + 1
            # print(i)

    # plt.figure()
    # plt.figure(figsize=(15, 10))
    # plt.plot(tmp1[:].reshape(-1), label='GroundTruth')
    # plt.plot(tmp2[:].reshape(-1), label='Prediction')
    # plt.legend()
    # plt.show()

counter5 = counter1 + counter4

"""
第二种情况：错误预报
在12个点内，真实值大于0.5的值 && 预测值小于0.5的值
此时，计数器加1
"""
counter2 = 0
for i, j in zip(range(trues.shape[0]), range(preds.shape[0])):
    tmp1 = trues[i, :, :]
    tmp2 = preds[j, :, :]
    if any(itr1 > 0.5 for itr1 in tmp1) and all(itr2 < 0.5 for itr2 in tmp2):
        counter2 = counter2 + 1
        # print(i)
    # plt.figure()
    # plt.figure(figsize=(15, 10))
    # plt.plot(tmp1[:].reshape(-1), label='GroundTruth')
    # plt.plot(tmp2[:].reshape(-1), label='Prediction')
    # plt.legend()
    # plt.show()

"""
第三种情况：虚假预报
在12个点内，真实值小于0.5的值 && 预测值大于0.5的值
此时，计数器加1
"""
counter3 = 0
for i, j in zip(range(trues.shape[0]), range(preds.shape[0])):
    tmp1 = trues[18060, :, :]
    tmp2 = preds[18060, :, :]
    if all(itr1 < 0.5 for itr1 in tmp1) and any(itr2 > 0.5 for itr2 in tmp2):
        counter3 = counter3 + 1
    # print(i)
# plt.figure()
# plt.figure(figsize=(15, 10))
# plt.plot(tmp1[:].reshape(-1), label='GroundTruth')
# plt.plot(tmp2[:].reshape(-1), label='Prediction')
# plt.legend()
# plt.show()

print("正确预报组数:")
print(counter5)
print("错误预报组数:")
print(counter2)
print("虚假预报组数:")
print(counter3)

print("Successfully!")