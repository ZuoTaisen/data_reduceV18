# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 16:07:50 2026

@author: zuots
"""

# -*- coding: utf-8 -*-
"""
sans_overlap_correction.py
==========================
通过迭代求解 D(theta, lambda) 修正因子，
使 TOF-SANS 数据在不同波长和不同角度下的 I(q) 曲线重叠。

算法：交替迭代法 (Alternating Optimization)
  Step A: 给定 D, 估计 I_true(q) —— q-bin 内加权平均
  Step B: 给定 I_true(q), 更新 D(theta, lambda) = I_meas / I_true
  Step C: 对 D 做平滑正则化
  重复直到收敛。
"""

import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from collections import defaultdict
global tester

class OverlapCorrector:
    """
    求解 D(theta, lambda) 使得 I_meas / D 在 q 空间重叠。
    
    Parameters
    ----------
    data : ndarray, shape (n_theta, n_lambda)
        测量强度 I(theta, lambda)，已做透射归一化。
    theta_array : ndarray, shape (n_theta,)
        散射角 2theta（弧度）。
    lambda_array : ndarray, shape (n_lambda,)
        波长数组（Å）。
    n_q_bins : int
        q 空间分 bin 数量。
    q_min, q_max : float or None
        q 范围，None 则自动确定。
    bs_points : int
        beam stop 遮挡的低角度像素数，跳过。
    """
    
    def __init__(self, data, theta_array, lambda_array, i0=None, lam_min = 6.0, lam_max = 10.5,
                 n_q_bins=50, q_min=None, q_max=None, bs_points=20):
        
        self.data = data.copy().astype(np.float64)
        self.theta = theta_array.copy()
        self.lam = lambda_array.copy()
        
        # i0: 立体角和入射波长谱的修正权重
        if i0 is not None:
            self.i0 = i0.copy().astype(np.float64)
        else:
            self.i0 = np.ones_like(self.data)
        
        self.n_theta = len(self.theta)
        self.n_lambda = len(self.lam)
        self.n_q_bins = n_q_bins
        self.bs_points = bs_points
        
        # 计算 q(theta, lambda) 矩阵
        # q = 4 * pi * sin(theta) / lambda  (这里 theta 是半角还是全角？)
        # 通常 SANS 中 theta_array 存的是 2theta，所以 sin(theta/2)
        # 但也可能直接是 theta。这里假设 theta_array 是 2theta。
        theta_2d = self.theta[:, np.newaxis]  # (n_theta, 1)
        lam_2d = self.lam[np.newaxis, :]      # (1, n_lambda)
        self.q_matrix = 4.0 * np.pi * np.sin(theta_2d / 2.0) / lam_2d
        
        # 有效像素 mask：排除 beam stop 和零/负值
        # 同时排除低计数点（计数 < 3）
        # i0 为 0 或负值也需要排除
        min_count_threshold = 3
        self.mask = np.ones((self.n_theta, self.n_lambda), dtype=bool)
        self.mask[:bs_points, :] = False
        self.mask[self.data <= 0] = False
        self.mask[self.data < min_count_threshold] = False
        self.mask[self.i0 <= 0] = False
        self.lam_min = lam_min
        self.lam_max = lam_max
        # 排除波长 < lam_min Å 和 > lam_max Å 的数据点
        lam_2d = self.lam[np.newaxis, :]  # shape: (1, n_lambda)
        lambda_mask = np.broadcast_to((lam_2d < self.lam_min) | (lam_2d > self.lam_max), self.mask.shape)
        n_lambda_excluded = np.sum(lambda_mask)
        self.mask[lambda_mask] = False
        print(f"[OverlapCorrector] Excluded {n_lambda_excluded} lambda points with lambda < 6.5 or > 9.7 Angstrom")
        
        # q 范围
        valid_q = self.q_matrix[self.mask]
        self.q_min = q_min if q_min is not None else np.percentile(valid_q, 1)
        self.q_max = q_max if q_max is not None else np.percentile(valid_q, 99)
        
        # 在 log 空间均匀分 bin
        self.q_bin_edges = np.logspace(np.log10(self.q_min), 
                                        np.log10(self.q_max), 
                                        n_q_bins + 1)
        self.q_bin_centers = np.sqrt(self.q_bin_edges[:-1] * self.q_bin_edges[1:])
        
        # 预计算每个像素属于哪个 q-bin
        self.q_bin_indices = np.digitize(self.q_matrix, self.q_bin_edges) - 1
        # 超出范围的标记为无效
        out_of_range = (self.q_bin_indices < 0) | (self.q_bin_indices >= n_q_bins)
        self.mask[out_of_range] = False
        self.q_bin_members = self._precompute_bin_mapping()
        
        print(f"[OverlapCorrector] Valid pixels: {self.mask.sum()} / {self.mask.size}")
        print(f"[OverlapCorrector] Q range: [{self.q_min:.5f}, {self.q_max:.5f}]")

    def _precompute_bin_mapping(self):
        """
        Precompute which Q bin each (theta, lambda) pixel belongs to.
        Excludes the first bs_points theta values (beamstop region).
        """
        # Digitize returns bin index (1 to n_q_bins), 0 or n_q_bins+1 for out-of-range
        bin_idx_raw = np.digitize(self.q_matrix, self.q_bin_edges)
        # Convert to 0-indexed, mark out-of-range as -1
        self.bin_indices = bin_idx_raw - 1
        out_of_range = (bin_idx_raw == 0) | (bin_idx_raw > self.n_q_bins)
        self.bin_indices[out_of_range] = -1
        
        # Build reverse mapping: bin -> list of pixel coordinates
        # Use self.mask for validity check (includes data > 0, data >= 3, i0 > 0, etc.)
        self.q_bin_members = defaultdict(list)
        ####################################################################################################
        for i_th in range(self.n_theta):
            for i_lam in range(self.n_lambda):
                if not self.mask[i_th, i_lam]:
                    continue
                b = self.bin_indices[i_th, i_lam]
                # 排除：Q范围外
                if b >= 0:
                    self.q_bin_members[b].append((i_th, i_lam))
        #################################################################################################
        print(f"  Q bin mapping: {len(self.q_bin_members)} bins with data")
        return self.q_bin_members
    
    def _estimate_i_true(self, corrected_data):
        """
        Step A: 在每个 q-bin 内对修正后的数据取加权平均，估计 I_true(q)。
        使用 1/I 作为权重（对数空间的均匀权重）。
        """
        i_true = np.zeros(self.n_q_bins)
        i_true_weight = np.zeros(self.n_q_bins)
        
        for i_th in range(self.n_theta):
            for i_lam in range(self.n_lambda):
                if not self.mask[i_th, i_lam]:
                    continue
                q_idx = self.q_bin_indices[i_th, i_lam]
                val = corrected_data[i_th, i_lam]
                if val > 0:
                    w = 1.0 / val  # log-space uniform weight
                    i_true[q_idx] += w * val
                    i_true_weight[q_idx] += w
        
        valid = i_true_weight > 0
        i_true[valid] /= i_true_weight[valid]
        i_true[~valid] = np.nan
        
        return i_true

    def _estimate_i_true_fast0(self, corrected_data):
        """
        Step A 的向量化版本，速度更快。
        """
        i_true = np.zeros(self.n_q_bins)
        counts = np.zeros(self.n_q_bins)
        log_sum = np.zeros(self.n_q_bins)
        
        # 取有效像素
        valid = self.mask & (corrected_data > 0)
        q_idx_flat = self.q_bin_indices[valid]
        val_flat = corrected_data[valid]
        log_val_flat = np.log(val_flat)
        
        # 在 log 空间取平均（几何平均），对 log-normal 分布更稳健
        np.add.at(log_sum, q_idx_flat, log_val_flat)
        np.add.at(counts, q_idx_flat, 1)
        
        valid_bins = counts > 0
        i_true[valid_bins] = np.exp(log_sum[valid_bins] / counts[valid_bins])
        i_true[~valid_bins] = np.nan

        return i_true

    def _estimate_i_true_fast(self, corrected_data):
        """
        Step A 的向量化版本，使用 q_bin_members 字典进行加权平均。
        对于每个 q-bin，利用该 bin 内所有像素的 i0 权重进行归一化加权平均。
        """
        i_true = np.zeros(self.n_q_bins)
        
        # 遍历每个 q-bin
        for b in range(self.n_q_bins):
            members = self.q_bin_members.get(b, [])
            if not members:
                i_true[b] = np.nan
                continue
            
            # 获取该 q-bin 内所有像素的 corrected_data 和 i0 值
            vals = np.array([corrected_data[i_th, i_lam] for i_th, i_lam in members])
            i0_vals = np.array([self.i0[i_th, i_lam] for i_th, i_lam in members])
            
            # 归一化权重：w = i0 / sum(i0)
            i0_sum = np.sum(i0_vals)
            if i0_sum > 0:
                weights = i0_vals / i0_sum
                # 加权平均：sum(val * w)
                i_true[b] = np.sum(vals * weights)
            else:
                i_true[b] = np.nan
        
        return i_true
    
    def _compute_d_raw(self, i_true):
        """
        Step B: D(theta, lambda) = I_meas(theta, lambda) / I_true(q(theta, lambda))
        """
        D = np.ones((self.n_theta, self.n_lambda))
        
        for i_th in range(self.n_theta):
            for i_lam in range(self.n_lambda):
                if not self.mask[i_th, i_lam]:
                    continue
                q_idx = self.q_bin_indices[i_th, i_lam]
                if np.isnan(i_true[q_idx]) or i_true[q_idx] <= 0:
                    continue
                D[i_th, i_lam] = self.data[i_th, i_lam] / i_true[q_idx]
        
        return D
    
    def _compute_d_raw_fast(self, i_true):
        """
        Step B 的向量化版本。
        """
        D = np.ones((self.n_theta, self.n_lambda))
        
        valid = self.mask.copy()
        q_idx = self.q_bin_indices.copy()
        q_idx[~valid] = 0  # 防止越界
        
        i_true_mapped = i_true[q_idx]
        
        # 排除 i_true 无效的 bin
        valid &= ~np.isnan(i_true_mapped)
        valid &= (i_true_mapped > 0)
        
        D[valid] = self.data[valid] / i_true_mapped[valid]
        
        return D
    
    def _smooth_d(self, D, sigma_theta=2.0, sigma_lambda=2.0):
        """
        Step C: 对 D 做高斯平滑正则化。
        只在有效区域内平滑，避免边界效应。
        """
        D_smooth = D.copy()
        
        # 用 mask 做加权平滑
        weight = self.mask.astype(np.float64)
        D_weighted = D_smooth * weight
        
        D_weighted_smooth = gaussian_filter(D_weighted, sigma=[sigma_theta, sigma_lambda])
        weight_smooth = gaussian_filter(weight, sigma=[sigma_theta, sigma_lambda])
        
        valid = weight_smooth > 1e-10
        D_smooth[valid] = D_weighted_smooth[valid] / weight_smooth[valid]
        D_smooth[~valid] = 1.0
        
        return D_smooth
    
    def solve(self, n_iter=0, sigma_theta=3.0, sigma_lambda=3.0, 
              verbose=True, tol=1e-4):
        """
        迭代求解 D(theta, lambda)。
        
        Parameters
        ----------
        n_iter : int
            最大迭代次数。
        sigma_theta : float
            D 在 theta 方向的高斯平滑宽度（像素）。
        sigma_lambda : float
            D 在 lambda 方向的高斯平滑宽度（像素）。
        verbose : bool
            是否打印迭代信息。
        tol : float
            收敛阈值（D 的相对变化）。
            
        Returns
        -------
        D : ndarray, shape (n_theta, n_lambda)
            修正因子。
        I_true : ndarray, shape (n_q_bins,)
            估计的真实 I(q)。
        history : list of float
            每次迭代的残差。
        """
        # 初始化 D = 1（无修正）
        D = np.ones((self.n_theta, self.n_lambda))
        history = []
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"  Iterative D(θ,λ) Solver")
            print(f"  sigma_theta={sigma_theta}, sigma_lambda={sigma_lambda}")
            print(f"{'='*60}")
        
        for iteration in range(n_iter):
            # Step A: 估计 I_true(q)
            corrected = self.data / D
            i_true = self._estimate_i_true_fast(corrected)
            
            # Step B: 计算原始 D
            D_new_raw = self._compute_d_raw_fast(i_true)
            
            # Step C: 平滑正则化
            D_new = self._smooth_d(D_new_raw, sigma_theta, sigma_lambda)
            
            # 归一化 D，使其中位数为 1（避免整体漂移）
            valid_d = D_new[self.mask]
            median_d = np.median(valid_d[valid_d > 0])
            if median_d > 0:
                D_new /= median_d
            
            # 计算收敛指标
            valid = self.mask & (D > 0) & (D_new > 0)
            rel_change = np.abs(D_new[valid] - D[valid]) / D[valid]
            mean_change = np.mean(rel_change)
            history.append(mean_change)
            
            if verbose:
                # 计算当前残差：修正后数据在 q-bin 内的离散度
                corrected_new = self.data / D_new
                scatter = self._compute_scatter(corrected_new)
                print(f"  Iter {iteration+1:3d}: "
                      f"mean|ΔD/D| = {mean_change:.6f}, "
                      f"scatter = {scatter:.6f}")
            
            D = D_new.copy()
            
            if mean_change < tol:
                if verbose:
                    print(f"  ✓ Converged at iteration {iteration+1}")
                break
        
        # 最终的 I_true
        corrected_final = self.data / D
        i_true_final = self._estimate_i_true_fast(corrected_final)
        
        self.D = D
        self.i_true = i_true_final
        self.history = history
        
        
        return D, i_true_final, history
    
    def _compute_scatter(self, corrected_data):
        """
        计算修正后数据在每个 q-bin 内的归一化标准差（对数空间），
        作为曲线重叠程度的度量。值越小越好。
        """
        valid = self.mask & (corrected_data > 0)
        q_idx_flat = self.q_bin_indices[valid]
        log_val_flat = np.log(corrected_data[valid])
        
        # 每个 bin 的均值和方差
        bin_sum = np.zeros(self.n_q_bins)
        bin_sum2 = np.zeros(self.n_q_bins)
        bin_count = np.zeros(self.n_q_bins)
        
        np.add.at(bin_sum, q_idx_flat, log_val_flat)
        np.add.at(bin_sum2, q_idx_flat, log_val_flat**2)
        np.add.at(bin_count, q_idx_flat, 1)
        
        valid_bins = bin_count > 1
        mean = bin_sum[valid_bins] / bin_count[valid_bins]
        var = bin_sum2[valid_bins] / bin_count[valid_bins] - mean**2
        var = np.maximum(var, 0)
        
        # 返回平均标准差
        return np.mean(np.sqrt(var))
    

    
    def get_corrected_iq_vs_lambda(self):
        """返回修正后的 I(q, lambda) 矩阵，使用 i0 加权平均"""
        corrected = self.data / self.D
        
        iq = np.zeros((self.n_q_bins, self.n_lambda))
        iq0 = np.zeros((self.n_q_bins, self.n_lambda))  # i0 权重累计
        
        valid = self.mask & (corrected > 0)
        
        for i_lam in range(self.n_lambda):
            for i_th in range(self.n_theta):
                if not valid[i_th, i_lam]:
                    continue
                q_idx = self.q_bin_indices[i_th, i_lam]
                # 使用 i0 加权：sum(corrected * i0) / sum(i0)
                iq[q_idx, i_lam] += corrected[i_th, i_lam] * self.i0[i_th, i_lam]
                iq0[q_idx, i_lam] += self.i0[i_th, i_lam]
        
        nonzero = iq0 > 0
        # 正确的归一化：每个 q-bin 分别除以该 bin 的 i0 总和
        iq[nonzero] /= iq0[nonzero]
        iq[~nonzero] = 0
        
        return self.q_bin_centers, iq

        
    
    def get_corrected_iq_vs_theta(self):
        """返回修正后的 I(q, theta) 矩阵，使用 i0 加权平均"""
        corrected = self.data / self.D
        
        iq = np.zeros((self.n_q_bins, self.n_theta))
        iq0 = np.zeros((self.n_q_bins, self.n_theta))  # i0 权重累计
        
        valid = self.mask & (corrected > 0)
        
        for i_th in range(self.n_theta):
            for i_lam in range(self.n_lambda):
                if not valid[i_th, i_lam]:
                    continue
                q_idx = self.q_bin_indices[i_th, i_lam]
                # 使用 i0 加权：sum(corrected * i0) / sum(i0)
                iq[q_idx, i_th] += corrected[i_th, i_lam] * self.i0[i_th, i_lam]
                iq0[q_idx, i_th] += self.i0[i_th, i_lam]
        
        nonzero = iq0 > 0
        # 正确的归一化：每个 q-bin 分别除以该 bin 的 i0 总和
        iq[nonzero] /= iq0[nonzero]
        iq[~nonzero] = 0
        
        return self.q_bin_centers, iq

def DLambda(data,theta,lam,i0,lam_min,lam_max,bs_points):
    corrector = OverlapCorrector(
        data=data / i0,  # 用 i0 归一化
        theta_array=theta,
        lambda_array=lam,
        i0=i0,  # 传入 i0 用于加权平均
        lam_min = lam_min,
        lam_max = lam_max,
        n_q_bins=100,
        q_min=None,
        q_max=None,
        bs_points=bs_points
    )
    
    D, i_true, history = corrector.solve(
        n_iter=3,
        sigma_theta=3.0,
        sigma_lambda=3.0,
        verbose=False,#True,
        tol=1e-5
    )
    # ========== 3. 获取修正前后的数据 ==========
    # 修正前（用原始 corrector 的方法，或者直接重新 bin）
    q_bins_before, iq_lam_before = _bin_raw_iq_vs_lambda_fast(corrector)
    q_bins_before, iq_th_before = _bin_raw_iq_vs_theta_fast(corrector)
    
    # 修正后
    q_bins_after, iq_lam_after = corrector.get_corrected_iq_vs_lambda()
    q_bins_after, iq_th_after = corrector.get_corrected_iq_vs_theta()
    
    # ========== 4. 绘图对比 ==========
    print("\nPlotting comparison...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # --- I(q, lambda) 修正前 ---
    ax = axes[0, 0]
    n_lam_plot = 12
    lam_indices = np.linspace(0, corrector.n_lambda - 1, n_lam_plot, dtype=int)
    colors = plt.cm.viridis(np.linspace(0, 1, n_lam_plot))
    
    for idx, i_lam in enumerate(lam_indices):
        valid = iq_lam_before[:, i_lam] > 0
        if valid.sum() > 0:
            ax.plot(q_bins_before[valid], iq_lam_before[valid, i_lam],
                    'o-', color=colors[idx], markersize=3, linewidth=1.2,
                    label=f'λ={corrector.lam[i_lam]:.1f}Å')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Q (Å⁻¹)'); ax.set_ylabel('I(Q, λ)')
    ax.set_title('BEFORE Correction: I(Q, λ)')
    ax.legend(fontsize=7, loc='upper right'); ax.grid(True, alpha=0.3)
    
    # --- I(q, lambda) 修正后 ---
    ax = axes[0, 1]
    for idx, i_lam in enumerate(lam_indices):
        valid = iq_lam_after[:, i_lam] > 0
        if valid.sum() > 0:
            ax.plot(q_bins_after[valid], iq_lam_after[valid, i_lam],
                    'o-', color=colors[idx], markersize=3, linewidth=1.2,
                    label=f'λ={corrector.lam[i_lam]:.1f}Å')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Q (Å⁻¹)'); ax.set_ylabel('I(Q, λ)')
    ax.set_title('AFTER Correction: I(Q, λ)')
    ax.legend(fontsize=7, loc='upper right'); ax.grid(True, alpha=0.3)
    
    # --- I(q, theta) 修正前 ---
    ax = axes[1, 0]
    n_th_plot = 15
    th_indices = np.unique(np.logspace(
        np.log10(corrector.bs_points), 
        np.log10(corrector.n_theta - 1), n_th_plot
    ).astype(int))
    colors2 = plt.cm.plasma(np.linspace(0, 1, len(th_indices)))
    
    for idx, i_th in enumerate(th_indices):
        valid = iq_th_before[:, i_th] > 0
        if valid.sum() > 0:
            ax.plot(q_bins_before[valid], iq_th_before[valid, i_th],
                    'o-', color=colors2[idx], markersize=3, linewidth=1.2,
                    label=f'θ={np.rad2deg(corrector.theta[i_th]):.2f}°')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Q (Å⁻¹)'); ax.set_ylabel('I(Q, θ)')
    ax.set_title('BEFORE Correction: I(Q, θ)')
    ax.legend(fontsize=7, loc='upper right'); ax.grid(True, alpha=0.3)
    
    # --- I(q, theta) 修正后 ---
    ax = axes[1, 1]
    for idx, i_th in enumerate(th_indices):
        valid = iq_th_after[:, i_th] > 0
        if valid.sum() > 0:
            ax.plot(q_bins_after[valid], iq_th_after[valid, i_th],
                    'o-', color=colors2[idx], markersize=3, linewidth=1.2,
                    label=f'θ={np.rad2deg(corrector.theta[i_th]):.2f}°')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Q (Å⁻¹)'); ax.set_ylabel('I(Q, θ)')
    ax.set_title('AFTER Correction: I(Q, θ)')
    ax.legend(fontsize=7, loc='upper right'); ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('correction_comparison.png', dpi=150, bbox_inches='tight')
    print("   Saved: correction_comparison.png")
    plt.show()
    
    # ========== 5. 绘制 D(theta, lambda) 和收敛曲线 ==========
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    
    # D 热图
    ax = axes2[0]
    im = ax.imshow(D, aspect='auto', origin='lower', cmap='RdBu_r',
                   vmin=np.percentile(D[corrector.mask], 5),
                   vmax=np.percentile(D[corrector.mask], 95))
    ax.set_xlabel('Lambda Index'); ax.set_ylabel('Theta Index')
    ax.set_title('D(θ, λ) Correction Factor')
    plt.colorbar(im, ax=ax, label='D')
    
    # D 的 lambda 依赖（对 theta 取平均）
    ax = axes2[1]
    D_vs_lam = np.nanmean(np.where(corrector.mask, D, np.nan), axis=0)
    ax.plot(corrector.lam, D_vs_lam, 'b.-', linewidth=1.5)
    ax.set_xlabel('Wavelength (Å)'); ax.set_ylabel('<D>(λ)')
    ax.set_title('D Averaged over θ')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    
    # 收敛曲线
    ax = axes2[2]
    ax.semilogy(range(1, len(history)+1), history, 'ko-', markersize=4)
    ax.set_xlabel('Iteration'); ax.set_ylabel('Mean |ΔD/D|')
    ax.set_title('Convergence History')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('correction_diagnostics.png', dpi=150, bbox_inches='tight')
    print("   Saved: correction_diagnostics.png")
    plt.show()
    
    # ========== 6. 最终合并的 I(q) ==========
    fig3, ax3 = plt.subplots(figsize=(10, 7))
    valid_iq = ~np.isnan(i_true) & (i_true > 0)
    ax3.plot(corrector.q_bin_centers[valid_iq], i_true[valid_iq], 
             'ko-', markersize=5, linewidth=2, label='I_true(q) from D-correction')
    ax3.set_xscale('log'); ax3.set_yscale('log')
    ax3.set_xlabel('Q (Å⁻¹)', fontsize=12)
    ax3.set_ylabel('I(Q)', fontsize=12)
    ax3.set_title('Final Merged I(Q) After D(θ,λ) Correction', fontsize=14)
    ax3.legend(); ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('final_merged_iq.png', dpi=150, bbox_inches='tight')
    print("   Saved: final_merged_iq.png")
    plt.show()
    
    print("\nAll done!")
    return D, i_true  

def main():
    """完整的修正流程演示"""
   
    import os, sys
    MODULE_DIR = os.path.join(os.path.dirname(__file__))
    sys.path.insert(0, MODULE_DIR)
    os.chdir(MODULE_DIR)
    
    # ========== 1. 加载数据 ==========
    print("Loading data...")
    data = np.load('SampleTransNormed.npy')
    i0 = np.load('Normalization.npy')
    theta = np.load('ThetaArray.npy')
    lam = np.load('WavelengthArray.npy')
    # global tester
    # tester = i0
    print(f"   Data: {data.shape}, Theta: {theta.shape}, Lambda: {lam.shape}")
    
    # ========== 2. 创建修正器并求解 ==========
    # i0 是立体角和入射波长谱的修正权重
    # 先对 data 进行归一化：data / i0
    corrector = OverlapCorrector(
        data=data / i0,  # 用 i0 归一化
        theta_array=theta,
        lambda_array=lam,
        i0=i0,  # 传入 i0 用于加权平均
        lam_min = 6.0,
        lam_max = 10.5,
        n_q_bins=100,
        q_min=None,
        q_max=None,
        bs_points=150
    )
    
    D, i_true, history = corrector.solve(
        n_iter=3,
        sigma_theta=3.0,
        sigma_lambda=3.0,
        verbose=False,#True,
        tol=1e-5
    )

    
    # ========== 3. 获取修正前后的数据 ==========
    # 修正前（用原始 corrector 的方法，或者直接重新 bin）
    q_bins_before, iq_lam_before = _bin_raw_iq_vs_lambda_fast(corrector)
    q_bins_before, iq_th_before = _bin_raw_iq_vs_theta_fast(corrector)
    
    # 修正后
    q_bins_after, iq_lam_after = corrector.get_corrected_iq_vs_lambda()
    q_bins_after, iq_th_after = corrector.get_corrected_iq_vs_theta()
    
    # ========== 4. 绘图对比 ==========
    print("\nPlotting comparison...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # --- I(q, lambda) 修正前 ---
    ax = axes[0, 0]
    n_lam_plot = 12
    lam_indices = np.linspace(0, corrector.n_lambda - 1, n_lam_plot, dtype=int)
    colors = plt.cm.viridis(np.linspace(0, 1, n_lam_plot))
    
    for idx, i_lam in enumerate(lam_indices):
        valid = iq_lam_before[:, i_lam] > 0
        if valid.sum() > 0:
            ax.plot(q_bins_before[valid], iq_lam_before[valid, i_lam],
                    'o-', color=colors[idx], markersize=3, linewidth=1.2,
                    label=f'λ={corrector.lam[i_lam]:.1f}Å')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Q (Å⁻¹)'); ax.set_ylabel('I(Q, λ)')
    ax.set_title('BEFORE Correction: I(Q, λ)')
    ax.legend(fontsize=7, loc='upper right'); ax.grid(True, alpha=0.3)
    
    # --- I(q, lambda) 修正后 ---
    ax = axes[0, 1]
    for idx, i_lam in enumerate(lam_indices):
        valid = iq_lam_after[:, i_lam] > 0
        if valid.sum() > 0:
            ax.plot(q_bins_after[valid], iq_lam_after[valid, i_lam],
                    'o-', color=colors[idx], markersize=3, linewidth=1.2,
                    label=f'λ={corrector.lam[i_lam]:.1f}Å')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Q (Å⁻¹)'); ax.set_ylabel('I(Q, λ)')
    ax.set_title('AFTER Correction: I(Q, λ)')
    ax.legend(fontsize=7, loc='upper right'); ax.grid(True, alpha=0.3)
    
    # --- I(q, theta) 修正前 ---
    ax = axes[1, 0]
    n_th_plot = 15
    th_indices = np.unique(np.logspace(
        np.log10(corrector.bs_points), 
        np.log10(corrector.n_theta - 1), n_th_plot
    ).astype(int))
    colors2 = plt.cm.plasma(np.linspace(0, 1, len(th_indices)))
    
    for idx, i_th in enumerate(th_indices):
        valid = iq_th_before[:, i_th] > 0
        if valid.sum() > 0:
            ax.plot(q_bins_before[valid], iq_th_before[valid, i_th],
                    'o-', color=colors2[idx], markersize=3, linewidth=1.2,
                    label=f'θ={np.rad2deg(corrector.theta[i_th]):.2f}°')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Q (Å⁻¹)'); ax.set_ylabel('I(Q, θ)')
    ax.set_title('BEFORE Correction: I(Q, θ)')
    ax.legend(fontsize=7, loc='upper right'); ax.grid(True, alpha=0.3)
    
    # --- I(q, theta) 修正后 ---
    ax = axes[1, 1]
    for idx, i_th in enumerate(th_indices):
        valid = iq_th_after[:, i_th] > 0
        if valid.sum() > 0:
            ax.plot(q_bins_after[valid], iq_th_after[valid, i_th],
                    'o-', color=colors2[idx], markersize=3, linewidth=1.2,
                    label=f'θ={np.rad2deg(corrector.theta[i_th]):.2f}°')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Q (Å⁻¹)'); ax.set_ylabel('I(Q, θ)')
    ax.set_title('AFTER Correction: I(Q, θ)')
    ax.legend(fontsize=7, loc='upper right'); ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('correction_comparison.png', dpi=150, bbox_inches='tight')
    print("   Saved: correction_comparison.png")
    plt.show()
    
    # ========== 5. 绘制 D(theta, lambda) 和收敛曲线 ==========
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    
    # D 热图
    ax = axes2[0]
    im = ax.imshow(D, aspect='auto', origin='lower', cmap='RdBu_r',
                   vmin=np.percentile(D[corrector.mask], 5),
                   vmax=np.percentile(D[corrector.mask], 95))
    ax.set_xlabel('Lambda Index'); ax.set_ylabel('Theta Index')
    ax.set_title('D(θ, λ) Correction Factor')
    plt.colorbar(im, ax=ax, label='D')
    
    # D 的 lambda 依赖（对 theta 取平均）
    ax = axes2[1]
    D_vs_lam = np.nanmean(np.where(corrector.mask, D, np.nan), axis=0)
    ax.plot(corrector.lam, D_vs_lam, 'b.-', linewidth=1.5)
    ax.set_xlabel('Wavelength (Å)'); ax.set_ylabel('<D>(λ)')
    ax.set_title('D Averaged over θ')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    
    # 收敛曲线
    ax = axes2[2]
    ax.semilogy(range(1, len(history)+1), history, 'ko-', markersize=4)
    ax.set_xlabel('Iteration'); ax.set_ylabel('Mean |ΔD/D|')
    ax.set_title('Convergence History')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('correction_diagnostics.png', dpi=150, bbox_inches='tight')
    print("   Saved: correction_diagnostics.png")
    plt.show()
    
    # ========== 6. 最终合并的 I(q) ==========
    fig3, ax3 = plt.subplots(figsize=(10, 7))
    valid_iq = ~np.isnan(i_true) & (i_true > 0)
    ax3.plot(corrector.q_bin_centers[valid_iq], i_true[valid_iq], 
             'ko-', markersize=5, linewidth=2, label='I_true(q) from D-correction')
    ax3.set_xscale('log'); ax3.set_yscale('log')
    ax3.set_xlabel('Q (Å⁻¹)', fontsize=12)
    ax3.set_ylabel('I(Q)', fontsize=12)
    ax3.set_title('Final Merged I(Q) After D(θ,λ) Correction', fontsize=14)
    ax3.legend(); ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('final_merged_iq.png', dpi=150, bbox_inches='tight')
    print("   Saved: final_merged_iq.png")
    plt.show()
    
    print("\nAll done!")
    
    return corrector, D, i_true

def _bin_raw_iq_vs_lambda(corrector):
    """对原始（未修正）数据做 q-binning，按 lambda 分组，使用 i0 加权平均"""
    iq = np.zeros((corrector.n_q_bins, corrector.n_lambda))
    iq0 = np.zeros((corrector.n_q_bins, corrector.n_lambda))  # i0 权重累计
    
    for i_th in range(corrector.n_theta):
        for i_lam in range(corrector.n_lambda):
            if not corrector.mask[i_th, i_lam]:
                continue
            if corrector.data[i_th, i_lam] <= 0:
                continue
            q_idx = corrector.q_bin_indices[i_th, i_lam]
            # 使用 i0 加权：sum(data * i0) / sum(i0)
            iq[q_idx, i_lam] += corrector.data[i_th, i_lam] * corrector.i0[i_th, i_lam]
            iq0[q_idx, i_lam] += corrector.i0[i_th, i_lam]
    
    nonzero = iq0 > 0
    iq[nonzero] /= iq0[nonzero]
    iq[~nonzero] = 0
    
    return corrector.q_bin_centers, iq


def _bin_raw_iq_vs_theta(corrector):
    """对原始（未修正）数据做 q-binning，按 theta 分组，使用 i0 加权平均"""
    iq = np.zeros((corrector.n_q_bins, corrector.n_theta))
    iq0 = np.zeros((corrector.n_q_bins, corrector.n_theta))  # i0 权重累计
    
    for i_th in range(corrector.n_theta):
        for i_lam in range(corrector.n_lambda):
            if not corrector.mask[i_th, i_lam]:
                continue
            if corrector.data[i_th, i_lam] <= 0:
                continue
            q_idx = corrector.q_bin_indices[i_th, i_lam]
            # 使用 i0 加权：sum(data * i0) / sum(i0)
            iq[q_idx, i_th] += corrector.data[i_th, i_lam] * corrector.i0[i_th, i_lam]
            iq0[q_idx, i_th] += corrector.i0[i_th, i_lam]
    
    nonzero = iq0 > 0
    iq[nonzero] /= iq0[nonzero]
    iq[~nonzero] = 0
    
    return corrector.q_bin_centers, iq

def _bin_raw_iq_vs_lambda_fast(corrector):
    """向量化版本，速度更快，使用 i0 加权平均"""
    iq = np.zeros((corrector.n_q_bins, corrector.n_lambda))
    iq0 = np.zeros((corrector.n_q_bins, corrector.n_lambda))  # i0 权重累计
    
    valid = corrector.mask & (corrector.data > 0)
    th_idx, lam_idx = np.where(valid)
    q_idx = corrector.q_bin_indices[th_idx, lam_idx]
    vals = corrector.data[th_idx, lam_idx]
    i0_vals = corrector.i0[th_idx, lam_idx]
    
    # 使用 i0 加权：sum(data * i0) / sum(i0)
    np.add.at(iq, (q_idx, lam_idx), vals * i0_vals)
    np.add.at(iq0, (q_idx, lam_idx), i0_vals)
    
    nonzero = iq0 > 0
    iq[nonzero] /= iq0[nonzero]
    iq[~nonzero] = 0
    
    return corrector.q_bin_centers, iq


def _bin_raw_iq_vs_theta_fast(corrector):
    """向量化版本，速度更快，使用 i0 加权平均"""
    iq = np.zeros((corrector.n_q_bins, corrector.n_theta))
    iq0 = np.zeros((corrector.n_q_bins, corrector.n_theta))  # i0 权重累计
    
    valid = corrector.mask & (corrector.data > 0)
    th_idx, lam_idx = np.where(valid)
    q_idx = corrector.q_bin_indices[th_idx, lam_idx]
    vals = corrector.data[th_idx, lam_idx]
    i0_vals = corrector.i0[th_idx, lam_idx]
    
    # 使用 i0 加权：sum(data * i0) / sum(i0)
    np.add.at(iq, (q_idx, th_idx), vals * i0_vals)
    np.add.at(iq0, (q_idx, th_idx), i0_vals)
    
    nonzero = iq0 > 0
    iq[nonzero] /= iq0[nonzero]
    iq[~nonzero] = 0
    
    return corrector.q_bin_centers, iq


if __name__ == "__main__":
    corrector, D, i_true = main()
    
'''
tt = D
theta = corrector.theta
lam = corrector.lam
for i in range(len(lam)):
    plt.plot(lam,tt[i,:])
    plt.yscale('log')
plt.show()
plt.close()
for i in range(len(lam)):
    plt.plot(theta,tt[:,i])
    plt.yscale('log')
plt.show()
'''