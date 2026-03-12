"""
inelastic_correction.py
=======================
Iterative self-consistency correction for TOF-SANS data.

This module implements a weighted iterative algorithm to correct for
inelastic/incoherent scattering effects, ensuring that normalized
intensities from different (theta, lambda) combinations mapping to
the same Q value converge to a consistent value.

Key Features:
- Uses raw counts as weights (higher counts = more reliable, less change)
- Iterative multiplicative update with damping for stability
- Comprehensive visualization and diagnostics

Algorithm:
----------
For each Q bin, we want all contributing (theta, lambda) pixels to have
the same normalized intensity. The update rule is:

    D_new[i,j] = D_old[i,j] * (1 + eta * (I_target - I_current) / I_current)

where:
- eta is a damping factor (smaller for high-count pixels)
- I_target is the weighted mean of the Q bin
- I_current is the current normalized intensity at pixel (i,j)

Author: Algorithm Engineer
Date: 2024
"""

import numpy as np
from collections import defaultdict
from typing import Tuple, Dict, Optional
import matplotlib.pyplot as plt
global tester

class InelasticCorrector:
    """
    Iterative self-consistency corrector for TOF-SANS inelastic scattering.
    """
    
    def __init__(
        self,
        data: np.ndarray,
        theta_array: np.ndarray,
        lambda_array: np.ndarray,
        i0_lambda: np.ndarray,
        n_q_bins: int = 40,
        q_min: Optional[float] = None,
        q_max: Optional[float] = None,
        bs_points: int = 20
    ):
        """
        Initialize the corrector.
        
        Parameters
        ----------
        data : ndarray, shape (N_theta, N_lambda)
            Raw neutron counts at each (theta, lambda) pixel.
        theta_array : ndarray, shape (N_theta,)
            Scattering angles in radians.
        lambda_array : ndarray, shape (N_lambda,)
            Neutron wavelengths (e.g., in Angstroms).
        i0_lambda : ndarray, shape (N_lambda,)
            Incident intensity spectrum I0(lambda) for normalization.
        n_q_bins : int
            Number of Q bins for mapping.
        q_min, q_max : float, optional
            Q range limits. If None, computed from data.
        bs_points : int
            Number of theta points at the beginning that are in the beamstop (excluded from Q binning).
        """
        # Store input data
        self.data = np.asarray(data, dtype=np.float64)
        self.theta = np.asarray(theta_array, dtype=np.float64)
        self.lam = np.asarray(lambda_array, dtype=np.float64)
        self.i0 = np.asarray(i0_lambda, dtype=np.float64)
        self.n_q_bins = n_q_bins
        self.bs_points = bs_points  # Beamstop points to exclude
        
        # Validate dimensions
        print(self.data.shape,'datashape',len(self.theta),len(self.lam))
        n_theta, n_lambda = self.data.shape
        if len(self.theta) != n_theta:
            raise ValueError(f"theta_array length {len(self.theta)} != data rows {n_theta}")
        if len(self.lam) != n_lambda:
            raise ValueError(f"lambda_array length {len(self.lam)} != data cols {n_lambda}")
        if len(self.i0[0]) != n_lambda:
            raise ValueError(f"i0_lambda length {len(self.i0)} != data cols {n_lambda}")
        
        self.n_theta = n_theta
        self.n_lambda = n_lambda
        
        # Validate bs_points
        if self.bs_points >= self.n_theta:
            raise ValueError(f"bs_points ({self.bs_points}) must be less than n_theta ({self.n_theta})")
        
        # Compute Q matrix: Q(theta, lambda) = 4*pi*sin(theta/2) / lambda
        sin_half_theta = np.sin(self.theta / 2.0)
        self.q_matrix = (4.0 * np.pi * sin_half_theta[:, np.newaxis] 
                         / self.lam[np.newaxis, :])
        
        # Setup Q bins
        self.q_min = 0.0015 #q_min if q_min is not None else np.nanmin(self.q_matrix)
        self.q_max = q_max if q_max is not None else np.nanmax(self.q_matrix)
        #self.q_bins = np.linspace(self.q_min, self.q_max, n_q_bins + 1)
        self.q_bins = np.logspace(np.log10(self.q_min), np.log10(self.q_max), n_q_bins + 1)
        self.q_bin_centers = 0.5 * (self.q_bins[:-1] + self.q_bins[1:])
        
        # Initialize correction factor matrix D(theta, lambda) = 1
        self.D = np.ones_like(self.data)
        
        # Precompute bin assignments
        self._precompute_bin_mapping()
        
        # History for convergence monitoring
        self.history = {
            'variation': [],
            'max_d_change': []
        }
    
    def _precompute_bin_mapping(self):
        """
        Precompute which Q bin each (theta, lambda) pixel belongs to.
        Excludes the first bs_points theta values (beamstop region).
        """
        global tester
        # Digitize returns bin index (1 to n_q_bins), 0 or n_q_bins+1 for out-of-range
        bin_idx_raw = np.digitize(self.q_matrix, self.q_bins)
        tester = bin_idx_raw
        # Convert to 0-indexed, mark out-of-range as -1
        self.bin_indices = bin_idx_raw - 1
        out_of_range = (bin_idx_raw == 0) | (bin_idx_raw > self.n_q_bins)
        self.bin_indices[out_of_range] = -1
        
        # Build reverse mapping: bin -> list of pixel coordinates
        # Exclude beamstop region (first bs_points theta indices)
        # Exclude points with count = 0 or count < min_count_threshold
        min_count_threshold = 3  # 最小计数阈值
        self.q_bin_members = defaultdict(list)
        ####################################################################################################
        for i_th in range(self.bs_points, self.n_theta):  # Skip first bs_points theta points
            for i_lam in range(self.n_lambda):
                b = self.bin_indices[i_th, i_lam]
                # 排除：Q范围外、计数为0、计数小于阈值
                if b >= 0 and self.data[i_th, i_lam] >= min_count_threshold:
                    self.q_bin_members[b].append((i_th, i_lam))
        #################################################################################################
        print(f"  Q bin mapping: {len(self.q_bin_members)} bins with data")
        print(f"  Excluded {self.bs_points} theta points (beamstop region)")
        print(f"  Excluded points with count = 0 or count < {min_count_threshold}")
    
    def compute_normalized_counts(self) -> np.ndarray:
        """
        Compute normalized counts: Data * D / I0(lambda).
        
        Returns
        -------
        norm_counts : ndarray, shape (N_theta, N_lambda)
        """
        # Avoid division by zero
        i0_safe = np.where(self.i0 > 0, self.i0, 1.0)
        control = np.where(self.i0 <= 0, 0, 1.0)
        return self.data * self.D / i0_safe*control
    
    def compute_q_bin_statistics(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute weighted mean and std for each Q bin.
        Uses raw counts as weights.
        
        Returns
        -------
        mean_iq : ndarray, shape (n_q_bins,)
        std_iq : ndarray, shape (n_q_bins,)
        count_per_bin : ndarray, shape (n_q_bins,)
        """
        norm_counts = self.compute_normalized_counts()
        weights = self.data  # Raw counts as weights
        
        mean_iq = np.full(self.n_q_bins, np.nan)
        std_iq = np.full(self.n_q_bins, np.nan)
        count_per_bin = np.zeros(self.n_q_bins)
        
        for b in range(self.n_q_bins):
            members = self.q_bin_members.get(b, [])
            if len(members) == 0:
                continue
            
            # Extract values and weights
            vals = np.array([norm_counts[i, j] for i, j in members])
            wts = np.array([weights[i, j] for i, j in members])
            
            # Filter valid entries
            valid = (wts > 0) & np.isfinite(vals) & (vals > 0)
            if valid.sum() == 0:
                continue
            
            vals = vals[valid]
            wts = wts[valid]
            
            # Weighted mean
            w_sum = wts.sum()
            w_mean = np.dot(wts, vals) / w_sum
            
            # Weighted std
            w_var = np.dot(wts, (vals - w_mean) ** 2) / w_sum
            w_std = np.sqrt(w_var)
            
            mean_iq[b] = w_mean
            std_iq[b] = w_std
            count_per_bin[b] = w_sum
        
        return mean_iq, std_iq, count_per_bin
    
    def compute_variation_metric(self) -> float:
        """
        Compute overall variation metric (weighted average CV).
        """
        mean_iq, std_iq, counts = self.compute_q_bin_statistics()
        
        valid = (mean_iq > 0) & np.isfinite(std_iq) & (counts > 0)
        if valid.sum() == 0:
            return np.inf
        
        cv = std_iq[valid] / mean_iq[valid]
        weights = counts[valid]
        avg_cv = np.dot(weights, cv) / weights.sum()
        
        return avg_cv
    
    def iterate_once(
        self, 
        damping: float = 0.5, 
        weight_exponent: float = 0.5,
        min_counts: float = 1.0
    ) -> float:
        """
        Perform one iteration of the correction algorithm.
        
        Parameters
        ----------
        damping : float
            Base step size (0 < damping <= 1).
        weight_exponent : float
            Controls resistance of high-count pixels (0 to 1).
        min_counts : float
            Minimum counts to consider a pixel valid.
        
        Returns
        -------
        max_d_change : float
            Maximum relative change in D matrix.
        """
        norm_counts = self.compute_normalized_counts()
        mean_iq, _, _ = self.compute_q_bin_statistics()
        
        # Compute normalized weights for resistance calculation
        max_count = self.data.max()
        if max_count > 0:
            normalized_counts = self.data / max_count
        else:
            normalized_counts = np.zeros_like(self.data)
        
        # Prepare new D matrix
        D_new = self.D.copy()
        max_change = 0.0
        
        for b in range(self.n_q_bins):
            target = mean_iq[b]
            members = self.q_bin_members.get(b, [])
            
            for i_th, i_lam in members:
                raw_count = self.data[i_th, i_lam]
                current = norm_counts[i_th, i_lam]
                
                # Skip invalid pixels
                if raw_count < min_counts or current <= 0 or not np.isfinite(current):
                    continue
                
                # Calculate ratio
                ratio = target / current
                
                # Resistance based on counts: high counts -> high resistance -> small change
                resistance = normalized_counts[i_th, i_lam] ** weight_exponent
                
                # Effective damping: higher resistance means smaller effective damping
                effective_damping = damping * (1.0 - resistance * 0.5)
                effective_damping = max(effective_damping, 0.01)  # minimum damping
                
                # Multiplicative update
                update_factor = 1.0 + effective_damping * (ratio - 1.0)
                
                # Clamp to prevent extreme values
                update_factor = np.clip(update_factor, 0.7, 1.5)
                
                D_new[i_th, i_lam] *= update_factor
                
                # Track maximum change
                rel_change = abs(update_factor - 1.0)
                if rel_change > max_change:
                    max_change = rel_change
        
        # Normalize D to keep mean close to 1 (optional stability measure)
        valid_d = D_new[self.data > min_counts]
        if len(valid_d) > 0:
            d_mean = valid_d.mean()
            if d_mean > 0:
                D_new /= d_mean
        
        self.D = D_new
        return max_change
    
    def run(
        self,
        n_iterations: int = 50,
        damping: float = 0.5,
        weight_exponent: float = 0.5,
        convergence_threshold: float = 1e-4,
        min_counts: float = 1.0,
        verbose: bool = True
    ) -> dict:
        """
        Run the iterative correction algorithm.
        
        Parameters
        ----------
        n_iterations : int
            Maximum number of iterations.
        damping : float
            Step size control (0 < damping <= 1).
        weight_exponent : float
            Controls resistance of high-count pixels.
        convergence_threshold : float
            Stop if max D change falls below this.
        min_counts : float
            Minimum counts to consider a pixel valid.
        verbose : bool
            Print progress information.
        
        Returns
        -------
        result : dict
            Contains 'D', 'I_Q', 'Q', 'history', etc.
        """
        if verbose:
            print("=" * 65)
            print("  Inelastic Scattering Correction - Iterative Algorithm")
            print("=" * 65)
            print(f"  Data shape: {self.data.shape}")
            print(f"  Q range: [{self.q_min:.4f}, {self.q_max:.4f}]")
            print(f"  Q bins: {self.n_q_bins}")
            print(f"  Parameters: damping={damping}, weight_exp={weight_exponent}")
            print("-" * 65)
        
        # Initial variation
        initial_var = self.compute_variation_metric()
        self.history['variation'].append(initial_var)
        
        if verbose:
            print(f"  Initial variation (CV): {initial_var:.6f}")
            print("-" * 65)
        
        for i in range(n_iterations):
            # Perform one iteration
            max_change = self.iterate_once(damping, weight_exponent, min_counts)
            
            # Compute new variation
            variation = self.compute_variation_metric()
            
            # Store history
            self.history['variation'].append(variation)
            self.history['max_d_change'].append(max_change)
            
            if verbose and (i + 1) % 5 == 0:
                print(f"  Iter {i+1:3d}: CV = {variation:.6f}, "
                      f"max_D_change = {max_change:.6f}")
            
            # Check convergence
            if max_change < convergence_threshold:
                if verbose:
                    print(f"  Converged at iteration {i+1}!")
                break
        
        # Final statistics
        final_var = self.compute_variation_metric()
        mean_iq, std_iq, counts = self.compute_q_bin_statistics()
        
        improvement = (1.0 - final_var / initial_var) * 100 if initial_var > 0 else 0
        
        if verbose:
            print("-" * 65)
            print(f"  Final variation (CV): {final_var:.6f}")
            print(f"  Improvement: {improvement:.2f}%")
            print(f"  D range: [{self.D.min():.4f}, {self.D.max():.4f}]")
            print("=" * 65)
        
        return {
            'D': self.D.copy(),
            'I_Q': mean_iq,
            'I_Q_std': std_iq,
            'Q': self.q_bin_centers.copy(),
            'counts_per_bin': counts,
            'history': self.history,
            'initial_variation': initial_var,
            'final_variation': final_var,
            'improvement_percent': improvement
        }
    
    def get_corrected_data(self) -> np.ndarray:
        """Return the corrected data matrix: Data * D."""
        return self.data * self.D
    
    def get_final_iq(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the final I(Q) curve.
        
        Returns
        -------
        q : ndarray
            Q values (bin centers).
        iq : ndarray
            Mean I(Q) values.
        iq_err : ndarray
            Standard deviation of I(Q).
        """
        mean_iq, std_iq, _ = self.compute_q_bin_statistics()
        return self.q_bin_centers.copy(), mean_iq, std_iq
    
    def get_iq_vs_lambda(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算 I(q, lambda) 矩阵：将相同 theta 索引对应的 data/i0 值按 Q-bin 叠加。
        
        Returns
        -------
        q_bins_center : ndarray
            Q bin 中心值，shape (n_q_bins,)
        iq_lambda : ndarray, shape (n_q_bins, n_lambda)
            I(q, lambda) 矩阵
        """
        # 计算归一化数据
        i0_safe = np.where(self.i0 > 0, self.i0, 1.0)
        norm_data = self.data / i0_safe
        
        # 初始化累加器和计数
        iq_lambda_sum = np.zeros((self.n_q_bins, self.n_lambda))
        iq_lambda_count = np.zeros((self.n_q_bins, self.n_lambda))
        
        # 遍历所有 Q bin 成员，按 theta 索引分组累加
        for b in range(self.n_q_bins):
            members = self.q_bin_members.get(b, [])
            for i_th, i_lam in members:
                iq_lambda_sum[b, i_lam] += norm_data[i_th, i_lam]
                iq_lambda_count[b, i_lam] += 1
        
        # 计算平均值，避免除零
        iq_lambda = np.divide(iq_lambda_sum, iq_lambda_count, 
                              out=np.zeros_like(iq_lambda_sum), 
                              where=iq_lambda_count > 0)
        
        return self.q_bin_centers.copy(), iq_lambda
    
    def get_iq_vs_theta(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算 I(q, theta) 矩阵：将相同 lambda 索引对应的 data/i0 值按 Q-bin 叠加。
        
        Returns
        -------
        q_bins_center : ndarray
            Q bin 中心值，shape (n_q_bins,)
        iq_theta : ndarray, shape (n_q_bins, n_theta)
            I(q, theta) 矩阵
        """
        # 计算归一化数据
        i0_safe = np.where(self.i0 > 0, self.i0, 1.0)
        norm_data = self.data / i0_safe
        
        # 初始化累加器和计数
        iq_theta_sum = np.zeros((self.n_q_bins, self.n_theta))
        iq_theta_count = np.zeros((self.n_q_bins, self.n_theta))
        
        # 遍历所有 Q bin 成员，按 lambda 索引分组累加
        for b in range(self.n_q_bins):
            members = self.q_bin_members.get(b, [])
            for i_th, i_lam in members:
                iq_theta_sum[b, i_th] += norm_data[i_th, i_lam]
                iq_theta_count[b, i_th] += 1
        
        # 计算平均值，避免除零
        iq_theta = np.divide(iq_theta_sum, iq_theta_count, 
                            out=np.zeros_like(iq_theta_sum), 
                            where=iq_theta_count > 0)
        
        return self.q_bin_centers.copy(), iq_theta


# ==========================================================================
# Visualization Functions
# ==========================================================================

def plot_results(corrector: InelasticCorrector, save_path: str = "correction_results.png"):
    """
    Generate comprehensive visualization of correction results.
    """

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # --- Plot 1: I(Q) before and after (top row, 2 columns) ---
    ax1 = fig.add_subplot(gs[0, :2])
    
    norm_before = corrector.data / corrector.i0 #np.where(corrector.i0[0] > 0, corrector.i0[0], 1.0)[np.newaxis, :]
    norm_after = corrector.compute_normalized_counts()

    n_lam_plot = min(8, corrector.n_lambda)
    lam_indices = np.linspace(0, corrector.n_lambda - 1, n_lam_plot, dtype=int)
    colors = plt.cm.viridis(np.linspace(0, 1, n_lam_plot))
    
    for idx, i_lam in enumerate(lam_indices):
        q_vals = corrector.q_matrix[:, i_lam]
        # 排除beamstop区域：前bs_points个theta点不显示
        valid = (corrector.data[:, i_lam] > 0) & (np.arange(corrector.n_theta) >= corrector.bs_points)
        
        if valid.sum() > 0:
            # Before (faded)
            ax1.scatter(q_vals[valid], norm_before[valid, i_lam], 
                        s=8, alpha=0.25, color=colors[idx])
            # After (solid)
            ax1.scatter(q_vals[valid], norm_after[valid, i_lam],
                        s=12, alpha=0.8, color=colors[idx],
                        label=f'lambda={corrector.lam[i_lam]:.1f}A')
    
    ax1.set_xlabel('Q (1/A)', fontsize=11)
    ax1.set_ylabel('Normalized I(Q)', fontsize=11)
    ax1.set_title('I(Q) vs Q: Before (faded) / After (solid) Correction', fontsize=12)
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.legend(fontsize=8, ncol=2, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # --- Plot 2: Convergence history (top right) ---
    ax2 = fig.add_subplot(gs[0, 2])
    iterations = range(len(corrector.history['variation']))
    ax2.plot(iterations, corrector.history['variation'], 'b-o', markersize=4, linewidth=1.5)
    ax2.set_xlabel('Iteration', fontsize=11)
    ax2.set_ylabel('Coefficient of Variation', fontsize=11)
    ax2.set_title('Convergence History', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # --- Plot 3: D matrix heatmap (middle left) ---
    ax3 = fig.add_subplot(gs[1, 0])
    d_range = max(abs(corrector.D.min() - 1), abs(corrector.D.max() - 1), 0.3)
    im3 = ax3.imshow(corrector.D, aspect='auto', origin='lower',
                     cmap='RdBu_r', vmin=1-d_range, vmax=1+d_range)
    ax3.set_xlabel('Lambda index', fontsize=11)
    ax3.set_ylabel('Theta index', fontsize=11)
    ax3.set_title('Correction Factor D(θ, λ)', fontsize=12)
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    # --- Plot 4: Original data heatmap (middle center) ---
    ax4 = fig.add_subplot(gs[1, 1])
    data_plot = np.log10(corrector.data + 1)
    im4 = ax4.imshow(data_plot, aspect='auto', origin='lower', cmap='viridis')
    ax4.set_xlabel('Lambda index', fontsize=11)
    ax4.set_ylabel('Theta index', fontsize=11)
    ax4.set_title('Original Data log10(counts+1)', fontsize=12)
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    
    # --- Plot 5: Q matrix heatmap (middle right) ---
    ax5 = fig.add_subplot(gs[1, 2])
    im5 = ax5.imshow(corrector.q_matrix, aspect='auto', origin='lower', cmap='plasma')
    ax5.set_xlabel('Lambda index', fontsize=11)
    ax5.set_ylabel('Theta index', fontsize=11)
    ax5.set_title('Q(θ, λ) Matrix', fontsize=12)
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    
    # --- Plot 6: Final I(Q) curve (bottom row, full width) ---
    ax6 = fig.add_subplot(gs[2, :])
    q, iq, iq_err = corrector.get_final_iq()
    valid = np.isfinite(iq) & (iq > 0)
    #print('IQ_Valid:',iq[valid])
    ax6.errorbar(q[valid], iq[valid], yerr=iq_err[valid], 
                fmt='o-', markersize=4, capsize=2, linewidth=1.5,
                color='steelblue', ecolor='gray', label='Corrected I(Q)')
    ax6.set_xlabel('Q (1/A)', fontsize=11)
    ax6.set_ylabel('I(Q)', fontsize=11)
    ax6.set_title('Final Corrected I(Q) Curve with Error Bars', fontsize=12)
    ax6.set_yscale('log')
    ax6.set_xscale('log')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Inelastic Scattering Correction Results', fontsize=14, fontweight='bold')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")
    plt.show()
    
    return fig


def plot_q_bin_detail(
    corrector: InelasticCorrector, 
    q_bin_index: Optional[int] = None,
    save_path: str = "q_bin_detail.png"
):
    """
    Plot detailed view of a single Q bin showing wavelength dependence.
    """
    # Find a Q bin with many members if not specified
    if q_bin_index is None:
        member_counts = {b: len(m) for b, m in corrector.q_bin_members.items()}
        if not member_counts:
            print("No Q bins with data!")
            return None
        q_bin_index = max(member_counts, key=member_counts.get)
    
    members = corrector.q_bin_members.get(q_bin_index, [])
    if len(members) == 0:
        print(f"Q bin {q_bin_index} has no members!")
        return None
    
    q_val = corrector.q_bin_centers[q_bin_index]
    
    # Extract data
    i0_safe = np.where(corrector.i0[0] > 0, corrector.i0[0], 1.0)
    norm_before = corrector.data / i0_safe[np.newaxis, :]
    norm_after = corrector.compute_normalized_counts()
    
    lambdas, thetas = [], []
    before_vals, after_vals, weights = [], [], []
    
    for i_th, i_lam in members:
        if corrector.data[i_th, i_lam] > 0:
            lambdas.append(corrector.lam[i_lam])
            thetas.append(corrector.theta[i_th])
            before_vals.append(norm_before[i_th, i_lam])
            after_vals.append(norm_after[i_th, i_lam])
            weights.append(corrector.data[i_th, i_lam])
    
    if len(lambdas) == 0:
        print(f"Q bin {q_bin_index} has no valid data points!")
        return None
    
    lambdas = np.array(lambdas)
    before_vals = np.array(before_vals)
    after_vals = np.array(after_vals)
    weights = np.array(weights)
    
    # Statistics
    mean_before = np.average(before_vals, weights=weights)
    mean_after = np.average(after_vals, weights=weights)
    std_before = np.sqrt(np.average((before_vals - mean_before)**2, weights=weights))
    std_after = np.sqrt(np.average((after_vals - mean_after)**2, weights=weights))
    cv_before = std_before / mean_before if mean_before > 0 else 0
    cv_after = std_after / mean_after if mean_after > 0 else 0
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Before
    ax1 = axes[0]
    sc1 = ax1.scatter(lambdas, before_vals, c=weights, cmap='viridis',
                      s=60, alpha=0.8, edgecolors='black', linewidths=0.5)
    ax1.axhline(mean_before, color='red', linestyle='--', linewidth=2,
                label=f'Mean = {mean_before:.4e}')
    ax1.axhspan(mean_before - std_before, mean_before + std_before,
                alpha=0.2, color='red')
    ax1.set_xlabel('Wavelength (A)', fontsize=11)
    ax1.set_ylabel('Normalized I(Q)', fontsize=11)
    ax1.set_title(f'Before Correction\nQ = {q_val:.4f}, CV = {cv_before:.4f}', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    plt.colorbar(sc1, ax=ax1, label='Raw Counts')
    
    # After
    ax2 = axes[1]
    sc2 = ax2.scatter(lambdas, after_vals, c=weights, cmap='viridis',
                      s=60, alpha=0.8, edgecolors='black', linewidths=0.5)
    ax2.axhline(mean_after, color='red', linestyle='--', linewidth=2,
                label=f'Mean = {mean_after:.4e}')
    ax2.axhspan(mean_after - std_after, mean_after + std_after,
                alpha=0.2, color='red')
    ax2.set_xlabel('Wavelength (A)', fontsize=11)
    ax2.set_ylabel('Normalized I(Q)', fontsize=11)
    ax2.set_title(f'After Correction\nQ = {q_val:.4f}, CV = {cv_after:.4f}', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    plt.colorbar(sc2, ax=ax2, label='Raw Counts')
    
    plt.suptitle(f'Q Bin {q_bin_index} Detail Analysis', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Q bin detail saved to: {save_path}")
    plt.show()
    
    return fig


# ==========================================================================
# Test Data Generation
# ==========================================================================

# ==========================================================================
# Test Data Generation
# ==========================================================================

def generate_test_data(
    n_theta: int = 50,
    n_lambda: int = 40,
    theta_range: Tuple[float, float] = (0.01, 0.15),
    lambda_range: Tuple[float, float] = (2.0, 12.0),
    inelastic_strength: float = 0.2,
    base_counts: float = 500.0,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic TOF-SANS data with simulated inelastic effects.
    
    Parameters
    ----------
    n_theta : int
        Number of theta bins.
    n_lambda : int
        Number of wavelength bins.
    theta_range : tuple
        (min, max) scattering angle in radians.
    lambda_range : tuple
        (min, max) wavelength in Angstroms.
    inelastic_strength : float
        Strength of simulated inelastic effect (0 to 1).
    base_counts : float
        Base count level for scaling.
    seed : int
        Random seed.
    
    Returns
    -------
    data : ndarray (n_theta, n_lambda)
    theta_array : ndarray (n_theta,)
    lambda_array : ndarray (n_lambda,)
    i0_lambda : ndarray (n_lambda,)
    """
    rng = np.random.default_rng(seed)
    
    theta_array = np.linspace(theta_range[0], theta_range[1], n_theta)
    lambda_array = np.linspace(lambda_range[0], lambda_range[1], n_lambda)
    
    # Incident spectrum: Maxwell-Boltzmann-like
    i0_lambda = lambda_array ** 2 * np.exp(-lambda_array / 4.0)
    i0_lambda = i0_lambda / i0_lambda.max() * base_counts
    
    # True I(Q) function
    def true_iq(q):
        return 10.0 / (q ** 2 + 0.005)
    
    # Compute Q matrix
    sin_half_theta = np.sin(theta_array / 2.0)
    q_matrix = 4.0 * np.pi * sin_half_theta[:, np.newaxis] / lambda_array[np.newaxis, :]
    
    # Generate data with inelastic effect
    data = np.zeros((n_theta, n_lambda))
    
    # Normalized lambda for inelastic effect calculation
    lam_min, lam_max = lambda_range
    lam_normalized = (lambda_array - lam_min) / (lam_max - lam_min)
    
    # Normalized theta for angle-dependent effect
    th_min, th_max = theta_range
    th_normalized = (theta_array - th_min) / (th_max - th_min)
    
    for i_th in range(n_theta):
        for i_lam in range(n_lambda):
            q = q_matrix[i_th, i_lam]
            
            # True scattering intensity
            intensity = true_iq(q)
            
            # Inelastic distortion factor:
            # - Longer wavelengths have larger apparent intensity drop
            # - Smaller angles are more affected
            lam_factor = 1.0 / (1.0 + inelastic_strength * lam_normalized[i_lam] ** 1.5)
            theta_factor = 1.0 - 0.3 * inelastic_strength * (1.0 - th_normalized[i_th])
            
            distorted_intensity = intensity * lam_factor * theta_factor
            
            # Expected counts = I(Q) * I0(lambda)
            expected_counts = distorted_intensity * i0_lambda[i_lam]
            
            # Add Poisson noise
            if expected_counts > 0:
                data[i_th, i_lam] = rng.poisson(expected_counts)
            else:
                data[i_th, i_lam] = 0
    i0_lambda = np.ones_like(theta_array[:,None])*i0_lambda
    
    return data, theta_array, lambda_array, i0_lambda


# ==========================================================================
# Main Demo Function
# ==========================================================================

if __name__ == "__main__":
    """
    Main demonstration: generate test data, run correction, visualize results.
    """
    print("\n")
    print("╔" + "═" * 62 + "╗")
    print("║" + " " * 62 + "║")
    print("║" + "  TOF-SANS Inelastic Scattering Correction Demo".center(62) + "║")
    print("║" + " " * 62 + "║")
    print("╚" + "═" * 62 + "╝")

    # -------------------------------------------------------------------------
    # Step 1: Generate synthetic test data
    # -------------------------------------------------------------------------
    print("\n[Step 1] Generating synthetic test data with inelastic effects...")

# data, theta, lam, i0 = generate_test_data(
#     n_theta=400,
#     n_lambda=250,
#     theta_range=(0.01, 5),
#     lambda_range=(2.0, 6.7),
#     inelastic_strength=0.25,
#     base_counts=800.0,
#     seed=42
# )

# print(f"         Data shape: {data.shape}")
# print(f"         Theta range: [{theta[0]:.4f}, {theta[-1]:.4f}] rad")
# print(f"         Lambda range: [{lam[0]:.1f}, {lam[-1]:.1f}] A")
# print(f"         Total counts: {data.sum():.0f}")
# print(f"         Non-zero pixels: {(data > 0).sum()}")

# -------------------------------------------------------------------------
# Step 2: Initialize the corrector
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n[Step 2] Initializing InelasticCorrector...")
    data = np.load('SampleTransNormed.npy')
    theta = np.load('ThetaArray.npy')
    lam = np.load('WavelengthArray.npy')
    i0 = np.load('Normalization.npy')
    corrector = InelasticCorrector(
        data=data,
        theta_array=theta,
        lambda_array=lam,
        i0_lambda=i0,
        n_q_bins=120,
        q_min=None,
        q_max=None,
        bs_points=20  # 排除beamstop区域的前20个theta点
    )

    # -------------------------------------------------------------------------
    # Step 3: Run iterative correction
    # -------------------------------------------------------------------------
    print("\n[Step 3] Running iterative correction algorithm...")

    result = corrector.run(
        n_iterations=10,
        damping=0.5,
        weight_exponent=0.5,
        convergence_threshold=1e-5,
        min_counts=1.0,
        verbose=True
    )

    # -------------------------------------------------------------------------
    # Step 4: Generate visualizations
    # -------------------------------------------------------------------------
    print("\n[Step 4] Generating visualizations...")

    # Main results plot
    plot_results(corrector, save_path="inelastic_correction_results.png")

    # Q bin detail plot
    plot_q_bin_detail(corrector, q_bin_index=None, save_path="q_bin_detail.png")

    # -------------------------------------------------------------------------
    # Step 5: Print summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"  Initial variation (CV): {result['initial_variation']:.6f}")
    print(f"  Final variation (CV):   {result['final_variation']:.6f}")
    print(f"  Improvement:            {result['improvement_percent']:.2f}%")
    print(f"  D matrix range:         [{result['D'].min():.4f}, {result['D'].max():.4f}]")
    print(f"  D matrix mean:          {result['D'].mean():.4f}")
    print(f"  D matrix std:           {result['D'].std():.4f}")
    print("=" * 65)

    # -------------------------------------------------------------------------
    # Step 6: Export final I(Q) curve
    # -------------------------------------------------------------------------
    print("\n[Step 6] Exporting final I(Q) curve...")

    q, iq, iq_err = corrector.get_final_iq()
    valid = np.isfinite(iq) & (iq > 0)

    # Save to file
    output_data = np.column_stack([q[valid], iq[valid], iq_err[valid]])
    np.savetxt("corrected_iq.dat", output_data, fmt="%.6e")
    print("         Saved to: corrected_iq.dat")

    print("\nDemo completed successfully!\n")

# tt = result['D']
# theta = corrector.theta
# lam = corrector.lam
# for i in range(len(lam)):
#     plt.plot(lam,tt[i,:])
#     plt.yscale('log')
#     plt.show()
# plt.close()
# for i in range(len(lam)):
#     plt.plot(theta,tt[:,i])
#     plt.yscale('log')
#     plt.show()

# return corrector, result


# ==========================================================================
# Entry Point
# ==========================================================================

# if __name__ == "__main__":
#     corrector, result = main()

