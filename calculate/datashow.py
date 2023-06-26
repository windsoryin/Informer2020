import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

setting = r'C:\Users\yuancao\Desktop\database\informer_JFNG_data_15min_ftMS_sl48_ll24_pl12_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0'

preds = np.load(setting+'/pred.npy')
trues = np.load(setting+'/true.npy')
# reals = np.load(setting+'/real_prediction_2023-04-22 17_00_57.797049.npy')

plt.figure()
plt.plot(trues[1,:,-1], label='GroundTruth')
plt.plot(preds[1,:,-1], label='Prediction')
plt.legend()
plt.show()

plt.figure()
plt.figure(figsize=(15, 10))
plt.plot(trues[:,:,-1].reshape(-1),label='GroundTruth')
plt.plot(preds[:,:,-1].reshape(-1),label='Prediction')
plt.legend()
plt.show()

# plt.figure()
# plt.plot(reals[0,:,-1], label='Prediction')
# plt.legend()
# plt.show()





