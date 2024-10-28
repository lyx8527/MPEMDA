import numpy as np
import torch

import dataload
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

from config_init import get_config

config = get_config()
device = config.device_topofallfeature



def fast_calculate_new(feature_matrix, neighbor_num):
    iteration_max = 10
    mu = 6
    X = feature_matrix.clone()
    alpha = torch.pow(X, 2).sum(axis=1)
    temp = alpha + alpha.unsqueeze(1) - 2 * torch.matmul(X, X.transpose(0, 1))
    temp[temp < 0] = 0
    distance_matrix = torch.sqrt(temp)
    row_num = X.shape[0]
    e = torch.ones((row_num, 1), device=X.device)
    distance_matrix = distance_matrix + torch.diag(torch.diag(e * e.transpose(0, 1)) * float('inf'))
    sort_index = torch.argsort(distance_matrix, dim=1)
    nearest_neighbor_index = sort_index[:, :neighbor_num].flatten()
    nearest_neighbor_matrix = torch.zeros((row_num, row_num), device=X.device)
    nearest_neighbor_matrix[torch.arange(row_num).repeat(neighbor_num), nearest_neighbor_index] = 1
    C = nearest_neighbor_matrix
    torch.manual_seed(0)
    W = torch.rand(row_num, row_num, device=X.device, dtype=torch.float)
    W = C * W
    lamda = mu * e
    P = torch.matmul(X, X.transpose(0, 1)) + torch.matmul(lamda, e.transpose(0, 1))
    P = P.to(W.dtype)
    for q in range(iteration_max):
        Q = torch.matmul(W, P)
        W = (W * P) / Q
        W = torch.nan_to_num(W)
    return W


def calculate_linear_neighbor_simi(feature_matrix, neighbor_rate):
    neighbor_num = int(neighbor_rate * feature_matrix.shape[0])
    return fast_calculate_new(feature_matrix, neighbor_num)


def normalize_by_divide_rowsum(simi_matrix):
    simi_matrix_copy = simi_matrix.clone()
    for i in range(simi_matrix_copy.shape[0]):
        simi_matrix_copy[i, i] = 0
    row_sum_matrix = simi_matrix_copy.sum(axis=1)
    result = simi_matrix_copy / row_sum_matrix.view(-1, 1)
    result[row_sum_matrix == 0, :] = 0
    return result


def complete_linear_neighbor_simi_matrix(train_association_matrix, neighbor_rate):
    b = torch.tensor(train_association_matrix)
    final_simi = calculate_linear_neighbor_simi(b, neighbor_rate)
    normalized_final_simi = normalize_by_divide_rowsum(final_simi)
    return normalized_final_simi


def linear_neighbor_predict(train_matrix, alpha):
    iteration_max = 1
    device = train_matrix.device
    drug_number = train_matrix.shape[0]
    microbe_number = train_matrix.shape[1]
    w_drug = complete_linear_neighbor_simi_matrix(train_matrix, 0.2)
    w_microbe = complete_linear_neighbor_simi_matrix(train_matrix.transpose(0, 1), 0.2)
    XX = []
    MS3 = dataload.sequencesim_microbe()
    XX.append(MS3)
    XX.append(w_microbe)
    MS = SNF(XX, 3, 5)
    YY = []
    DS2 = dataload.atcsim_drug()
    YY.append(DS2)
    YY.append(w_drug)
    DS = SNF(YY, 3, 5)
    M_Similaritys = []
    M_Similaritys.append(MS3)
    M_Similaritys.append(w_microbe)
    M_Similaritys.append(MS)
    S_Similaritys = []
    S_Similaritys.append(DS2)
    S_Similaritys.append(w_drug)
    S_Similaritys.append(DS)
    S1 = []
    w_drug = DS
    w_microbe = MS
    w_drug_eye = torch.eye(drug_number, device=device)
    w_microbe_eye = torch.eye(microbe_number, device=device)
    temp0 = w_drug_eye - alpha * w_drug

    for q in range(iteration_max):
        try:
            temp1 = torch.linalg.inv(temp0)
        except Exception:
            temp1 = torch.linalg.pinv(temp0)
        temp1 = temp1.to(train_matrix.dtype)
        temp2 = temp1 @ train_matrix
    prediction_drug = (1 - alpha) * temp2
    temp3 = w_microbe_eye - alpha * w_microbe

    for p in range(iteration_max):
        try:
            temp4 = torch.linalg.inv(temp3)
        except Exception:
            temp4 = torch.linalg.pinv(temp3)
        temp4 = temp4.to(train_matrix.dtype)
        temp5 = temp4 @ train_matrix.T
    temp6 = (1 - alpha) * temp5
    prediction_microbe = temp6.T
    prediction_result = 0.5 * prediction_drug + 0.5 * prediction_microbe
    return prediction_result


def Error_LPA(train_matrix, DS, MS, alpha):
    iteration_max = 1
    device = config.device_topofallfeature
    train_matrix_tensor = train_matrix.to(device)

    w_drug = DS.to(device)
    w_microbe = MS.to(device)
    drug_number = train_matrix.shape[0]
    microbe_number = train_matrix.shape[1]
    w_drug_eye = torch.eye(drug_number, device=device)
    w_microbe_eye = torch.eye(microbe_number, device=device)
    temp0 = (w_drug_eye - alpha * w_drug).to(device)

    for q in range(iteration_max):
        try:
            temp1 = torch.linalg.inv(temp0)
        except Exception:
            temp1 = torch.linalg.pinv(temp0)
        temp2 = (temp1 @ train_matrix_tensor).to(device)
    prediction_drug = (1 - alpha) * temp2
    temp3 = (w_microbe_eye - alpha * w_microbe).to(device)

    for p in range(iteration_max):
        try:
            temp4 = torch.linalg.inv(temp3)
        except Exception:
            temp4 = torch.linalg.pinv(temp3)
        temp5 = (temp4 @ train_matrix_tensor.T).to(device)
    temp6 = (1 - alpha) * temp5
    prediction_microbe = temp6.T
    prediction_result = 0.5 * prediction_drug + 0.5 * prediction_microbe
    return prediction_result.cpu().numpy()


def calculate_metrics(test, pred_probs, threshold=0.5):
    # 将概率转换为二分类标签
    pred_labels = (pred_probs >= threshold).astype(int)

    # 计算 AUC
    auc = roc_auc_score(test, pred_probs)

    # 计算 AUPR
    aupr = average_precision_score(test, pred_probs)

    # 计算 ACC
    acc = accuracy_score(test, pred_labels)

    return auc, aupr, acc, pred_labels


def find_best_threshold(test, pred_probs):
    thresholds = np.linspace(0, 1, 100)  # 选择100个阈值
    best_threshold = 0
    best_acc = 0

    for threshold in thresholds:
        _, _, acc, _ = calculate_metrics(test, pred_probs, threshold)
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold

    return best_threshold


def FindDominantSet(W, K):
    m, n = W.shape
    DS = torch.zeros((m, n), dtype=torch.float32)  # 修改数据类型为 torch.float32
    for i in range(m):
        index = torch.argsort(W[i, :])[-K:]  # 获取最近的 K 个邻居
        DS = torch.tensor(DS, device=device)
        W = torch.tensor(W, device=device)
        DS[i, index] = W[i, index]  # 保留最近的邻居

    # 按行归一化
    B = torch.sum(DS, dim=1, keepdim=True)
    DS = DS / B
    return DS


def normalized(W, ALPHA):
    m, n = W.shape
    W = W + ALPHA * torch.eye(m, device=device)
    return (W + W.t()) / 2


def SNF(Wall, K, t, ALPHA=1):
    for i in range(len(Wall)):
        Wall[i] = torch.tensor(Wall[i], device=device)
    C = len(Wall)
    m, n = Wall[0].shape


    for i in range(C):
        Wall[i] = torch.tensor(Wall[i], dtype=torch.float32, device=device)  # 将 NumPy 数组转换为 PyTorch 张量，同时指定数据类型为 torch.float32
        B = torch.sum(Wall[i], dim=1, keepdim=True)  # 在维度 1 上求和，并保持维度
        Wall[i] = Wall[i] / B
        Wall[i] = (Wall[i] + Wall[i].t()) / 2  # 转置矩阵并求和以得到对称矩阵

    newW = []

    for i in range(C):
        newW.append(FindDominantSet(Wall[i], K))

    Wsum = torch.zeros((m, n), device=device)
    for i in range(C):
        Wsum += Wall[i]

    for iteration in range(t):
        Wall0 = []
        for i in range(C):
            temp = torch.matmul(torch.matmul(newW[i], (Wsum - Wall[i])), newW[i].t()) / (C - 1)  # 使用 torch.matmul 进行矩阵乘法
            Wall0.append(temp)

        for i in range(C):
            Wall0[i] = torch.tensor(Wall0[i], device=device)
            Wall[i] = normalized(Wall0[i], ALPHA)

        Wsum = torch.zeros((m, n), device=device)
        for i in range(C):
            Wsum += Wall[i]

    W = Wsum / C
    B = torch.sum(W, dim=1, keepdim=True)  # 在维度 1 上求和，并保持维度
    W /= B
    W = (W + W.t() + torch.eye(m, device=device)) / 2  # 转置矩阵并求和以得到对称矩阵
    return W


def WKNKN(Y, SD, ST, K, eta):
    Yd = np.zeros(Y.shape)
    Yt = np.zeros(Y.shape)
    wi = np.zeros((K,))
    wj = np.zeros((K,))
    num_drugs, num_targets = Y.shape
    for i in np.arange(num_drugs):
        dnn_i = torch.argsort(torch.tensor(SD[i, :]), descending=True)[:K]  # 将 NumPy 数组转换为 PyTorch 张量，然后调用 torch.argsort
        Zd = torch.sum(torch.tensor(SD[i, dnn_i]))
        for ii in np.arange(K):
            wi[ii] = (eta ** ii) * SD[i, dnn_i[ii]]
        if not np.isclose(Zd, 0.):
            Yd[i, :] = np.sum(np.multiply(wi.reshape((K, 1)), Y[dnn_i, :]), axis=0) / Zd  # 对应元素相乘 按列求和
    for j in np.arange(num_targets):
        tnn_j = torch.argsort(torch.tensor(ST[j, :]), descending=True)[:K]  # 将 NumPy 数组转换为 PyTorch 张量，然后调用 torch.argsort
        Zt = torch.sum(torch.tensor(ST[j, tnn_j]))
        for jj in np.arange(K):
            wj[jj] = (eta ** jj) * ST[j, tnn_j[jj]]
        if not np.isclose(Zt, 0.):
            Yt[:, j] = np.sum(np.multiply(wj.reshape((1, K)), Y[:, tnn_j]), axis=1) / Zt
    Ydt = (Yd + Yt) / 2
    x, y = np.where(Ydt > Y)

    Y_tem = Y.copy()
    Y_tem[x, y] = Ydt[x, y]
    return Y_tem
