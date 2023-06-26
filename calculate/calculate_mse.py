import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

setting = r'C:\Users\yuancao\Desktop\database\informer_JFNG_data_15min_ftMS_sl48_ll24_pl12_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0'

trues=np.load(setting+'/true.npy',allow_pickle= True)
preds=np.load(setting+'/pred.npy',allow_pickle= True)

arr = trues-preds
# print(arr)
# print(len(arr))

mse1 = np.mean((preds - trues)**2)
print("----------没有删掉小于0.1的值的mse1----------")
print("mse1：",mse1)

true_length=len(trues)
pred_length=len(preds)
print("原始数据真实值长度：",true_length)
print("原始数据预测值长度：",pred_length)
print("------------------------------------------")

# true_reshape=trues.reshape(true_length*24,1)
# pred_reshape=preds.reshape(true_length*24,1)

# plt.figure()
# plt.title('picture_name')  # 图片标题
# plt.xlabel('x')  # x轴变量名称
# plt.ylabel('y')  # y轴变量名称
# plt.plot(true_reshape[1:10000,0],label='true')  # 画出 a_line 线  label="x": 图中左上角示例
# plt.plot(pred_reshape[1:10000,0],label='pred')  # 画出 a_line 线  label="x": 图中左上角示例
# plt.legend()  # 画出曲线图标
# plt.show()  # 画出图像


for i in range(trues.shape[0]):
    tmp=trues[i,:,:]
    if all(itr < 0.1 for itr in tmp):
        trues[i, :, :]= -9999

new_trues = trues[trues != -9999]
new_trues = new_trues.reshape(-1, 24, 1)
new_preds = preds[trues != -9999]
new_preds = new_preds.reshape(-1, 24, 1)

true_length=len(new_trues)
pred_length=len(new_preds)

print("------------删掉小于0.1的值的mse2------------")
mse2 = np.mean((new_preds - new_trues)**2)
print("删除小于0.1后数据真实值长度：",true_length)
print("删除小于0.1后数据预测值长度：",pred_length)
print("mse2：",mse2)


new_true_reshape=new_trues.reshape(true_length*24,1)
new_pred_reshape=new_preds.reshape(true_length*24,1)

# plt.figure()
# plt.title('picture_name')  # 图片标题
# plt.xlabel('x轴')  # x轴变量名称
# plt.ylabel('y轴')  # y轴变量名称
# plt.plot(new_true_reshape)  # 画出 a_line 线  label="x": 图中左上角示例
# plt.plot(new_pred_reshape)  # 画出 a_line 线  label="x": 图中左上角示例
# plt.legend()  # 画出曲线图标
# plt.show()  # 画出图像


