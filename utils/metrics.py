import numpy as np
# 6/20增加代码
from tslearn.metrics import dtw, dtw_path


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


# 6/20添加代码
# def DTW(pred, true):
#     distance, path = dtw_path(pred, true)
#     return distance

# 6/20添加代码
# def Temporal(pred, true):
#     tdi = 0
#     N_output = pred.shape[1]
#     distance, path = dtw_path(pred, true)
#     Dist = 0
#     for i, j in path:
#         Dist += (i - j) * (i - j)
#     tdi += Dist / (N_output * N_output)
#     return tdi

# 6/25 add code
def DTW(pred, true):
    losses_dtw = []
    losses_tdi = []

    loss_dtw, loss_tdi = torch.tensor(0), torch.tensor(0)
    batch_size, N_output = true.shape[0:2]

    loss_dtw, loss_tdi = 0, 0
    # DTW and TDI
    for k in range(batch_size):
        pred_k_cpu = pred[k, :, 0:1].view(-1).detach().cpu().numpy()
        true_k_cpu = true[k, :, 0:1].view(-1).detach().cpu().numpy()

        path, sim = dtw_path(pred_k_cpu, true_k_cpu)
        loss_dtw += sim

        Dist = 0
        for i, j in path:
            Dist += (i - j) * (i - j)
        loss_tdi += Dist / (N_output * N_output)

    loss_dtw = loss_dtw / batch_size
    loss_tdi = loss_tdi / batch_size

    # print statistics
    losses_dtw.append(loss_dtw)
    losses_tdi.append(loss_tdi)

    # print(' dtw= ', np.array(losses_dtw).mean(), ' tdi= ', np.array(losses_tdi).mean())

    return np.array(losses_dtw).mean()


def Temporal(pred, true):
    losses_dtw = []
    losses_tdi = []

    loss_dtw, loss_tdi = torch.tensor(0), torch.tensor(0)
    batch_size, N_output = true.shape[0:2]

    loss_dtw, loss_tdi = 0, 0
    # DTW and TDI
    for k in range(batch_size):
        pred_k_cpu = pred[k, :, 0:1].view(-1).detach().cpu().numpy()
        true_k_cpu = true[k, :, 0:1].view(-1).detach().cpu().numpy()

        path, sim = dtw_path(pred_k_cpu, true_k_cpu)
        loss_dtw += sim

        Dist = 0
        for i, j in path:
            Dist += (i - j) * (i - j)
        loss_tdi += Dist / (N_output * N_output)

    loss_dtw = loss_dtw / batch_size
    loss_tdi = loss_tdi / batch_size

    # print statistics
    losses_dtw.append(loss_dtw)
    losses_tdi.append(loss_tdi)

    # print(' dtw= ', np.array(losses_dtw).mean(), ' tdi= ', np.array(losses_tdi).mean())

    return np.array(losses_tdi).mean()


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    # 6/20增加代码
    dtw = DTW(pred, true)
    tdi = Temporal(pred, true)

    return mae, mse, rmse, mape, mspe, dtw, tdi