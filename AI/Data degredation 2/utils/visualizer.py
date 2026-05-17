import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class Visualizer:
    """
    Master PHM Diagnostic Engine v4.0.
    Implements 6-Panel Diagnostic Matrix with Research-Validated Overlay.
    Standards: IEEE PHM, Industry 4.0 Dashboarding.
    """
    def __init__(self, plot_dir: str = "plots"):
        self.plot_dir = plot_dir
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)
        
        # High-resolution aesthetic configuration
        plt.rcParams.update({
            'font.size': 10,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'figure.dpi': 200,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linestyle': '--'
        })

    def plot_local_diagnostic_dashboard(self, faulty_df: pd.DataFrame, healthy_df: pd.DataFrame, 
                                      feature_map: dict, joint_idx: int, 
                                      start_idx: int, duration: int, fault_name: str):
        """
        Elite 6-Panel Diagnostic Dashboard.
        Dual-Line Overlay: Gray (Baseline) vs. Red/Gradient (Degraded).
        """
        fig, axes = plt.subplots(3, 2, figsize=(18, 18))
        axes = axes.flatten()
        
        # Window Padding for context (800 samples)
        pad = 800
        w_start = max(0, start_idx - pad)
        w_end = min(len(faulty_df), start_idx + duration + pad)
        indices = faulty_df.index[w_start:w_end]
        
        c_h, c_f, c_z = '#7f8c8d', '#c0392b', '#e74c3c' # Gray, DarkRed, LightRed

        # PILLAR A: KINEMATIC PRECISION (Pos Error Residual)
        t_col = feature_map.get('target_q', [])[joint_idx]
        a_col = feature_map.get('actual_q', [])[joint_idx]
        if t_col in faulty_df.columns and a_col in faulty_df.columns:
            f_res = np.abs(faulty_df[t_col].iloc[indices] - faulty_df[a_col].iloc[indices])
            h_res = np.abs(healthy_df[t_col].iloc[indices] - healthy_df[a_col].iloc[indices])
            axes[0].plot(indices, h_res, color=c_h, alpha=0.5, label='Baseline Residual')
            axes[0].plot(indices, f_res, color=c_f, linewidth=1.5, label='Degraded Residual')
            axes[0].set_ylabel("Pos Error Δq [rad]")
            axes[0].set_yscale('log')
            axes[0].set_title("1. KINEMATIC PRECISION (RESIDUAL)")

        # PILLAR B: MECHANICAL STRESS (Motor Current)
        cur_col = feature_map.get('actual_current', [])[joint_idx]
        if cur_col in faulty_df.columns:
            axes[1].plot(indices, healthy_df[cur_col].iloc[indices], color=c_h, alpha=0.5, label='Baseline')
            axes[1].plot(indices, faulty_df[cur_col].iloc[indices], color=c_f, label='Degraded')
            axes[1].set_ylabel("Motor Current [A]")
            axes[1].set_title("2. MECHANICAL STRESS (CURRENT)")

        # PILLAR C: THERMAL HEALTH (Temperature)
        temp_col = feature_map.get('joint_temperature', [])[joint_idx]
        if temp_col in faulty_df.columns:
            axes[2].plot(indices, healthy_df[temp_col].iloc[indices], color=c_h, alpha=0.5, label='Baseline')
            axes[2].plot(indices, faulty_df[temp_col].iloc[indices], color='#d35400', linewidth=2, label='Degraded')
            axes[2].set_ylabel("Temperature [°C]")
            axes[2].set_title("3. THERMAL HEALTH (GRADIENT)")

        # PILLAR D: DYNAMIC STABILITY (Velocity)
        vel_col = feature_map.get('actual_qd', [])[joint_idx]
        if vel_col in faulty_df.columns:
            axes[3].plot(indices, healthy_df[vel_col].iloc[indices], color=c_h, alpha=0.5, label='Baseline')
            axes[3].plot(indices, faulty_df[vel_col].iloc[indices], color='#2980b9', label='Degraded')
            axes[3].set_ylabel("Velocity [rad/s]")
            axes[3].set_title("4. DYNAMIC STABILITY (JITTER)")

        # PILLAR E: CARTESIAN STABILITY (TCP Spatial Repeatability)
        tcp_z = [c for c in faulty_df.columns if 'tcp' in c.lower() and 'z' in c.lower()]
        if tcp_z:
            axes[4].plot(indices, healthy_df[tcp_z[0]].iloc[indices], color=c_h, alpha=0.5, label='Baseline')
            axes[4].plot(indices, faulty_df[tcp_z[0]].iloc[indices], color='#2c3e50', label='Degraded')
            axes[4].set_ylabel("TCP Z [m]")
            axes[4].set_title("5. CARTESIAN STABILITY (REPEATABILITY)")

        # PILLAR F: CBM DIAGNOSTIC OVERLAY (Combined Residual)
        if cur_col in faulty_df.columns:
            residual = np.abs(faulty_df[cur_col].iloc[indices] - healthy_df[cur_col].iloc[indices])
            axes[5].fill_between(indices, 0, residual, color='red', alpha=0.3, label='System Residual')
            axes[5].plot(indices, residual, color='darkred', linewidth=1)
            axes[5].set_ylabel("Residual ΔI [A]")
            axes[5].set_title("6. CBM DIAGNOSTIC RESIDUAL")

        # RESEARCH MODULE ANNOTATIONS & SYMBOLS
        anno_style = dict(color='white', weight='bold', bbox=dict(boxstyle="round,pad=0.3", fc=c_z, ec="none", alpha=0.8))
        
        if fault_name == 'FAULT_LF':
            axes[2].text(start_idx + duration/4, axes[2].get_ylim()[1]*0.8, "VISCOUS SHEAR DRIFT", **anno_style)
        elif fault_name == 'FAULT_BL':
            axes[1].annotate('RE-ENGAGEMENT STRIKE', xy=(start_idx + 10, faulty_df[cur_col].loc[start_idx+10]), 
                            xytext=(start_idx-200, faulty_df[cur_col].max()*0.9),
                            arrowprops=dict(facecolor='black', shrink=0.05), color='darkred', weight='bold')
        elif fault_name == 'FAULT_BW':
            axes[3].text(start_idx + duration/4, axes[3].get_ylim()[1]*0.7, "RCF VIBRATION ZONE", **anno_style)
        elif fault_name == 'FAULT_EE':
            axes[1].text(start_idx, axes[1].get_ylim()[1]*0.8, "ARCING SPIKES", color='magenta', weight='bold')
            axes[2].annotate('EDM FLUTING STEP', xy=(start_idx, faulty_df[temp_col].loc[start_idx]), 
                            xytext=(start_idx-300, faulty_df[temp_col].loc[start_idx]+10),
                            arrowprops=dict(arrowstyle="->", connectionstyle="angle3"))
        elif fault_name == 'FAULT_MD':
            axes[1].annotate('Kt DECAY GAP', xy=(start_idx + duration/2, faulty_df[cur_col].loc[start_idx+duration/2]), 
                            xytext=(start_idx, faulty_df[cur_col].max()*1.1),
                            arrowprops=dict(arrowstyle="fancy"))

        # Global Polish
        for ax in axes:
            ax.axvspan(start_idx, start_idx + duration, color=c_z, alpha=0.1, label='Active Fault')
            ax.legend(loc='upper right', fontsize=8)
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f"Master_Diagnostic_{fault_name}.png"))
        plt.close()

    def plot_stitching_dashboard(self, df, baseline_len, feature_map, joint_idx=1):
        """Visualizes the data expansion stitching quality."""
        plt.figure(figsize=(15, 6))
        col = feature_map['actual_q'][joint_idx]
        plt.plot(df[col], label='Expanded Signal', color='#2ecc71', alpha=0.8)
        plt.axvline(baseline_len, color='red', linestyle='--', label='Stitching Boundary')
        plt.title("PHM Digital Twin: 2x Expansion Stitching Audit")
        plt.xlabel("Sample Index"); plt.ylabel("Position [rad]")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "Expansion_Stitching_Audit.png"))
        plt.close()
