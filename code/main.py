import datetime
import sys
import warnings
import math
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import dataload
import utils

warnings.filterwarnings("ignore")
sys.setrecursionlimit(10000)

from config_init import get_config




def calculate(alpha, Times):
    print("正在执行模型计算，请稍等。")
    path = fr"../dataset/{dataset}"
    start = datetime.datetime.now()
    A, pp, m, n = dataload.Amatrix2()
    types_num = 1
    times_num = Times
    aucs = np.zeros((times_num, types_num))
    auprs = np.zeros((times_num, types_num))
    for times in range(Times):
        print(f"第 {times + 1} 次计算:")
        np.random.seed(np.random.randint(0, 100) + times)
        randlist = np.random.permutation(pp)
        pred = []
        cv_aucs = np.zeros((types_num, 5))
        cv_auprs = np.zeros((types_num, 5))
        for cv in range(5):
            print(f"Fold {cv + 1} of 5-fold cross-validation:")
            if cv != 4:
                partrandlist = randlist[cv * math.floor(pp / 5):(cv + 1) * math.floor(pp / 5)].copy()
            else:
                partrandlist = randlist[cv * math.floor(pp / 5):pp].copy()
            A2, rand_1 = dataload.A2matrix2_Drugvirus(A, partrandlist)
            indices = np.where(A2 == 1)
            num_indeices = len(indices[0])
            random_indices = np.random.permutation(num_indeices)
            split_indices = np.array_split(random_indices, 20)
            A_test = np.zeros_like(A2)
            for i in range(len(rand_1)):
                A_test[int(rand_1[i][0]), int(rand_1[i][1])] = 1
            XX = []
            YY = []
            MS = dataload.gausssim_microbe(A2, A2.shape[1])
            DS = dataload.gausssim_drug(A2, A2.shape[0])
            XX.append(MS)
            YY.append(DS)
            MS2 = np.loadtxt(path + '/Microbe_functional.txt', dtype=np.float32)
            DS2 = np.loadtxt(path + '/Drug_smile.txt', dtype=np.float32)
            XX.append(MS2)
            YY.append(DS2)
            SM = utils.SNF(XX, 3, 2)
            SD = utils.SNF(YY, 3, 2)
            M_Similarity = [torch.tensor(MS, device=device), torch.tensor(MS2, device=device),
                            torch.tensor(SM, device=device)]                                            # 1 gauss similarity  2 functional similarity  3 integrated similarity
            D_Similarity = [torch.tensor(DS, device=device), torch.tensor(DS2, device=device),
                            torch.tensor(SD, device=device)]                                            # 1 gauss similarity  2 smile similarity  3 integrated similarity
            A_mark = torch.zeros_like(torch.tensor(A2, device=device))
            for i, indices_chunk in enumerate(tqdm(split_indices, desc='Processing')):
                A3 = torch.clone(torch.tensor(A2, device=device))
                A3[indices[0][indices_chunk], indices[1][indices_chunk]] = 0
                A3__1 = utils.WKNKN(A3.cpu().numpy(), D_Similarity[1].cpu().numpy(), M_Similarity[1].cpu().numpy(), 20, 0.9)
                A3__2 = utils.WKNKN(A3.cpu().numpy(), D_Similarity[2].cpu().numpy(), M_Similarity[2].cpu().numpy(), 20, 0.9)
                A3 = torch.tensor(np.where(A3__1 > A3__2, A3__1, A3__2), device=device)
                A3 = utils.linear_neighbor_predict(A3, alpha=alpha)
                A_mark[indices[0][indices_chunk], indices[1][indices_chunk]] = A3[
                    indices[0][indices_chunk], indices[1][indices_chunk]]
            A3_1 = utils.WKNKN(torch.tensor(A2, device=device).cpu().numpy(), D_Similarity[2].cpu().numpy(),
                             M_Similarity[2].cpu().numpy(), 20, 0.9)                                     # 标准的为1 1
            A3_2 = utils.WKNKN(torch.tensor(A2, device=device).cpu().numpy(), D_Similarity[2].cpu().numpy(),
                             M_Similarity[2].cpu().numpy(), 20, 0.9)
            A3_3 = torch.tensor(np.where(A3_1 > A3_2, A3_1, A3_2), device=device)
            S1 = utils.linear_neighbor_predict(A3_3, alpha=alpha)
            A_mark1 = torch.clone(S1)
            A_mark1[indices[0], indices[1]] = A_mark[indices[0], indices[1]]
            for jj in range(0, len(rand_1)):
                A_mark1[int(rand_1[jj][0]), int(rand_1[jj][1])] = 0
            min_val = torch.min(A_mark1)
            max_val = torch.max(A_mark1)
            scale_factor = 1 / (max_val - min_val)
            normalized_A_mark1 = (A_mark1 - min_val) * scale_factor
            E_train = torch.tensor(A2, device=device) - normalized_A_mark1
            E_all = utils.Error_LPA(torch.tensor(E_train, dtype=torch.float32), D_Similarity[2], M_Similarity[2],
                                     alpha=alpha)
            E_all_tensor = torch.tensor(E_all, device=device, dtype=torch.float32)
            E_test = torch.zeros_like(E_train, device=device, dtype=torch.float32)
            for jj in range(0, len(rand_1)):
                E_test[int(rand_1[jj][0]), int(rand_1[jj][1])] = E_all_tensor[int(rand_1[jj][0]), int(rand_1[jj][1])]
            scaler = MinMaxScaler(feature_range=(-0.004, 0.004))
            scaled_E_test = scaler.fit_transform(E_test.cpu().numpy())
            S_all = [S1 + torch.tensor(scaled_E_test, device=device)]
            pos = len(partrandlist)
            y_true = np.array([0] * (len(rand_1) - pos) + [1] * pos)
            test = []
            y_pred = np.array([0] * (len(rand_1)))
            y_pred = y_pred.astype('float32')
            for c in range(len(S_all)):
                S = S_all[c]  # 将 PyTorch 张量移动到 CPU 并转换为 NumPy 数组
                for jj in range(0, len(rand_1)):
                    y_pred[jj] = y_pred[jj] + (S[int(rand_1[jj][0]), int(rand_1[jj][1])])
                pred = np.append(pred, y_pred)
                best_threshold = utils.find_best_threshold(y_true, y_pred)
                auc, aupr, acc, pred_labels = utils.calculate_metrics(y_true, y_pred, best_threshold)
                cv_aucs[c, cv] = auc
                cv_auprs[c, cv] = aupr

        # 将 pred 放到 GPU 上
        aucs[times, :] = np.mean(cv_aucs, axis=1)
        auprs[times, :] = np.mean(cv_auprs, axis=1)

        print("------------------")
    end = datetime.datetime.now()
    row_means = np.mean(aucs, axis=0)
    row_vars = np.var(aucs, axis=0)
    row_means1 = np.mean(auprs, axis=0)
    row_vars1 = np.var(auprs, axis=0)
    print(f"Mean AUC on {dataset} dataset:")
    print(np.array_str(row_means, precision=4) + " ± " + np.array_str(row_vars, precision=4))
    print(f"Mean AUPR on {dataset} dataset:")
    print(np.array_str(row_means1, precision=4) + " ± " + np.array_str(row_vars1, precision=4))
    print(end - start)


if __name__ == '__main__':
    alpha = 0.1
    Times = 1
    config = get_config()
    device = config.device_topofallfeature
    dataset = config.dataset_topofallfeature
    print("Using device:", device)

    # Move your alpha and Times variables to the GPU if available
    alpha = torch.tensor(alpha, device=device)
    Times = torch.tensor(Times, device=device)

    # Run calculate function on GPU
    calculate(alpha, Times)
