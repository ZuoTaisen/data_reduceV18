# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 18:22:02 2026

@author: zuots
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import mstats  # 加权中位数工具


# Q Bin划分
def create_q_bins(Q_matrix, num_bins=100):
    Q_min, Q_max = Q_matrix.min(), Q_matrix.max()
    # 使用log坐标划分Q bins
    bin_edges = np.logspace(np.log10(Q_min), np.log10(Q_max), num_bins+1)
    bin_indices = np.digitize(Q_matrix, bin_edges) - 1  # 0-based索引
    bin_indices = np.clip(bin_indices, 0, num_bins-1)  # 边界保护
    return bin_edges, bin_indices  # bin_indices: (N_theta, N_lambda)

def weighted_median(values, weights):
    """计算加权中位数"""
    sorted_idx = np.argsort(values)
    sorted_values = values[sorted_idx]
    sorted_weights = weights[sorted_idx]
    cum_weights = np.cumsum(sorted_weights)
    threshold = cum_weights[-1] / 2.0
    return sorted_values[np.searchsorted(cum_weights, threshold)]

def optimize_d_with_weighting(Data, Q_matrix, bin_indices, I0, num_bins,
                             max_iter=100, eps=1e-6, alpha=0.5, sigma_smooth=1.0):
    N_theta, N_lambda = Data.shape
    D = np.ones((N_theta, N_lambda))  # 初始D全1
    prev_loss = np.inf
    weights = Data  # 权重=原始计数（用户要求：计数越多权重越大）
    
    # 处理I0：支持一维向量(N_lambda,)或二维矩阵(N_theta, N_lambda)
    I0 = np.asarray(I0)
    if I0.ndim == 1:
        I0 = I0[None, :]  # 广播为 (1, N_lambda)
    elif I0.ndim == 2:
        pass  # 已经是二维矩阵
    
    for t in range(max_iter):
        # E步：修正数据、归一化计数、估计Bin目标强度A_k
        Count = Data * D  # 未归一计数
        Count_norm = Count / I0  # 归一化计数（仅使用I0，不做立体角修正）
        
        A_k = np.zeros(num_bins)  # 每个Bin的目标强度
        #print(A_k)
        for k in range(num_bins):
            mask = (bin_indices == k)
            if np.sum(mask) < 1:  # 跳过点数过少的Bin
                A_k[k] = 0
                continue
            vals = Count_norm[mask]
            ws = weights[mask]
            A_k[k] = weighted_median(vals, ws)  # 加权中位数估计
        
        # M步：阻尼更新D
        D_new = np.ones_like(D)
        for i in range(N_theta):
            for j in range(N_lambda):
                k = bin_indices[i, j]
                if A_k[k] <= 0 or Data[i, j] == 0:
                    D_new[i, j] = D[i, j]  # 保持原D
                else:
                    # 理论D值 = (A_k * I0) / Data（不做立体角修正）
                    # I0可以是二维矩阵(N_theta, N_lambda)或一维向量(N_lambda,)
                    D_theory = (A_k[k] * I0[i, j] if I0.ndim == 2 else A_k[k] * I0[j]) / Data[i, j]
                    # 阻尼更新：D_new = (1-alpha)*D_old + alpha*D_theory
                    D_new[i, j] = (1 - alpha) * D[i, j] + alpha * D_theory
        
        # 正则化：高斯平滑
        D_smooth = gaussian_filter(D_new, sigma=[sigma_smooth, sigma_smooth])
        D = D_smooth  # 更新D为平滑后的值
        
        # 计算加权损失函数（目标函数）
        loss = 0.0
        for k in range(num_bins):
            mask = (bin_indices == k)
            if np.sum(mask) < 1:
                continue
            vals = Count_norm[mask]  # 当前归一计数
            ws = weights[mask]  # 权重
            loss += np.sum(ws * (vals - A_k[k])**2)  # 加权方差和
        
        # 收敛判断
        if t > 0 and abs(prev_loss - loss) < eps:
            print(f"Converged at iter {t}, loss={loss:.2e}")
            break
        prev_loss = loss
    
    return D, Data * D  # 返回修正因子与修正后数据

if __name__=="__main__":
    # 输入数据（示例）
    N_theta, N_lambda = 100, 50  # 角度、波长点数
    Data = np.random.poisson(lam=100, size=(N_theta, N_lambda))  # 二维计数（泊松分布模拟）
    LambdaArray = np.linspace(0.5, 5.0, N_lambda)  # 波长向量（nm）
    ThetaArray = np.linspace(0.1, 5.0, N_theta)  # 角度向量（度）
    I0 = np.ones((N_theta, N_lambda)) * 1e6  # 入射通量：二维矩阵(N_theta, N_lambda)
    
    # 计算Q矩阵（二维）
    theta_rad = np.deg2rad(ThetaArray)
    Q_matrix = 4 * np.pi * np.sin(theta_rad[:, None]/2) / LambdaArray[None, :]  # shape (N_theta, N_lambda)
    
    bin_edges, bin_indices = create_q_bins(Q_matrix, num_bins=100)
    num_bins = len(bin_edges) - 1
    
    # 执行优化
    D_opt, Data_corrected = optimize_d_with_weighting(
        Data, Q_matrix, bin_indices, I0, num_bins,
        max_iter=50, alpha=0.3, sigma_smooth=1.5
    )
    
    # 验证：同一Q Bin内归一化计数的变异
    Count_corrected = Data_corrected
    Count_norm_corrected = Count_corrected / I0  # 仅使用I0归一化
    bin_vars = []
    for k in range(num_bins):
        mask = (bin_indices == k)
        if np.sum(mask) > 5:
            bin_vars.append(np.var(Count_norm_corrected[mask]))
    #print(f"Mean bin variance: {np.mean(bin_vars):.4e} (优化前: {np.mean(np.var(Count_norm, axis=1)):.4e})")