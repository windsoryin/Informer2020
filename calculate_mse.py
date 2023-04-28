import numpy as np

setting = r'C:\Users\yuancao\Desktop\JFNGdata\JFNGdata_result\vali_result\informer_Exp_JFNG_data_15min_ftMS_sl672_ll96_pl24_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_1'

trues=np.load(setting+'/vali_true.npy',allow_pickle= True)
preds=np.load(setting+'/vali_pred.npy',allow_pickle= True)

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


for i in range(trues.shape[0]):
    for j in range(trues.shape[1]):
        for k in range(trues.shape[2]):
            if trues[i][j][k] < 0.1:
                trues[i][j][k] = 0
                preds[i][j][k] = 0

new_trues = trues[trues != 0]
new_trues = new_trues.reshape(-1, 24, 1)
new_preds = preds[preds != 0]
new_preds = new_preds.reshape(-1, 24, 1)

true_length=len(new_trues)
pred_length=len(new_preds)

print("------------删掉小于0.1的值的mse2------------")
mse2 = np.mean((new_preds - new_trues)**2)
print("删除小于0.1后数据真实值长度：",true_length)
print("删除小于0.1后数据预测值长度：",pred_length)
print("mse2：",mse2)





