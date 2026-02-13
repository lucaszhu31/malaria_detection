"""
Compare CE FPN + CBAM Training Results vs Baseline

Analyzes training curves from:
1. yolo11m_cbam_sahi_1024px (baseline - 85% mAP)
2. yolo11m_ce_fpn_cbam_1024px (enhanced - target 88-90% mAP)
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt

def load_results(results_dir):
    """Load results.csv from training directory"""
    results_file = Path(results_dir) / 'results.csv'
    
    if not results_file.exists():
        print(f"⚠️  Results file not found: {results_file}")
        return None
    
    # Read CSV manually to preserve column structure
    data = {}
    with open(results_file) as f:
        header = f.readline().strip().split(',')
        header = [h.strip() for h in header]
        
        for col in header:
            data[col] = []
        
        for line in f:
            values = line.strip().split(',')
            for i, col in enumerate(header):
                try:
                    data[col].append(float(values[i]))
                except:
                    data[col].append(None)
    
    return data, header


def compare_training_runs(
    baseline_dir='/content/drive/MyDrive/kaggle_results/cbam_experiments/yolo11m_cbam_sahi_1024px',
    enhanced_dir='/content/drive/MyDrive/kaggle_results/cbam_experiments/yolo11m_ce_fpn_cbam_1024px',
    save_plot='/tmp/ce_fpn_vs_cbam_comparison.png',
):
    """Compare training curves between two experiments"""
    
    print("\n" + "="*70)
    print("Comparing CE FPN + CBAM vs CBAM-only Baseline")
    print("="*70 + "\n")
    
    # Load baseline
    print(f"📊 Loading baseline from: {baseline_dir}")
    baseline = load_results(baseline_dir)
    if baseline is None:
        print("❌ Baseline results not found. Run train_yolo_cbam.py first.")
        return
    baseline_data, baseline_header = baseline
    
    # Load enhanced
    print(f"📊 Loading CE FPN variant from: {enhanced_dir}")
    enhanced = load_results(enhanced_dir)
    if enhanced is None:
        print("⚠️  CE FPN results not yet available. Run train_yolo11m_ce_fpn_cbam.py first.")
        enhanced_data = None
    else:
        enhanced_data, enhanced_header = enhanced
    
    # Parse metrics
    print("\n" + "-"*70)
    print("FINAL METRICS (Last Epoch)")
    print("-"*70)
    
    # Find mAP columns
    map_col = None
    for col in baseline_header:
        if 'metrics/mAP50(B)' in col:
            map_col = col
            break
    
    if map_col is None:
        print("⚠️  Could not find mAP column. Available columns:")
        print(baseline_header)
        return
    
    baseline_final_map = baseline_data[map_col][-1] if baseline_data[map_col][-1] is not None else 'N/A'
    print(f"\n✅ CBAM-only (Baseline)")
    print(f"   Final mAP: {baseline_final_map:.1%}" if isinstance(baseline_final_map, float) else f"   Final mAP: {baseline_final_map}")
    
    if enhanced_data:
        enhanced_final_map = enhanced_data[map_col][-1] if enhanced_data[map_col][-1] is not None else 'N/A'
        print(f"\n🚀 CE FPN + CBAM (Enhanced)")
        print(f"   Final mAP: {enhanced_final_map:.1%}" if isinstance(enhanced_final_map, float) else f"   Final mAP: {enhanced_final_map}")
        
        if isinstance(baseline_final_map, float) and isinstance(enhanced_final_map, float):
            improvement = enhanced_final_map - baseline_final_map
            improvement_pct = (improvement / baseline_final_map) * 100
            print(f"\n📈 Improvement: +{improvement:.1%} ({improvement_pct:.1f}% relative)")
    
    print("-"*70)
    
    # Create comparison plot
    if enhanced_data and save_plot:
        print(f"\n📉 Creating comparison plot...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('CE FPN + CBAM vs CBAM-only Baseline Comparison', fontsize=14, fontweight='bold')
        
        # mAP comparison
        ax = axes[0, 0]
        epochs_baseline = list(range(1, len(baseline_data[map_col])+1))
        epochs_enhanced = list(range(1, len(enhanced_data[map_col])+1))
        
        ax.plot(epochs_baseline, baseline_data[map_col], 'o-', label='CBAM-only (Baseline)', linewidth=2)
        ax.plot(epochs_enhanced, enhanced_data[map_col], 's-', label='CE FPN + CBAM', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('mAP50')
        ax.set_title('Validation mAP Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Find training loss columns
        train_loss_col = None
        for col in baseline_header:
            if 'train/loss' in col and 'box' not in col:
                train_loss_col = col
                break
        
        if train_loss_col and train_loss_col in enhanced_header:
            ax = axes[0, 1]
            ax.plot(epochs_baseline, baseline_data[train_loss_col], 'o-', label='CBAM-only', linewidth=2)
            ax.plot(epochs_enhanced, enhanced_data[train_loss_col], 's-', label='CE FPN + CBAM', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Training Loss')
            ax.set_title('Training Loss Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Convergence speed
        ax = axes[1, 0]
        
        # Find when each reached 80% mAP
        baseline_80 = None
        enhanced_80 = None
        
        for i, v in enumerate(baseline_data[map_col]):
            if v is not None and v >= 0.80:
                baseline_80 = i + 1
                break
        
        for i, v in enumerate(enhanced_data[map_col]):
            if v is not None and v >= 0.80:
                enhanced_80 = i + 1
                break
        
        speeds = []
        labels = []
        if baseline_80:
            speeds.append(baseline_80)
            labels.append(f'CBAM-only\n(Epoch {baseline_80})')
        if enhanced_80:
            speeds.append(enhanced_80)
            labels.append(f'CE FPN+CBAM\n(Epoch {enhanced_80})')
        
        if speeds:
            ax.bar(labels, speeds, color=['#1f77b4', '#ff7f0e'], alpha=0.7)
            ax.set_ylabel('Epochs to 80% mAP')
            ax.set_title('Convergence Speed (Lower is Better)')
            ax.set_ylim(0, max(speeds) * 1.2)
        
        # mAP progression
        ax = axes[1, 1]
        
        milestone_epochs = [10, 20, 30, 40]
        baseline_milestones = []
        enhanced_milestones = []
        
        for ep in milestone_epochs:
            if ep <= len(baseline_data[map_col]):
                baseline_milestones.append(baseline_data[map_col][ep-1])
            if ep <= len(enhanced_data[map_col]):
                enhanced_milestones.append(enhanced_data[map_col][ep-1])
        
        x = range(len(milestone_epochs))
        width = 0.35
        
        if baseline_milestones:
            ax.bar([i - width/2 for i in x[:len(baseline_milestones)]], 
                   baseline_milestones, width, label='CBAM-only', alpha=0.8)
        if enhanced_milestones:
            ax.bar([i + width/2 for i in x[:len(enhanced_milestones)]], 
                   enhanced_milestones, width, label='CE FPN+CBAM', alpha=0.8)
        
        ax.set_ylabel('mAP50')
        ax.set_title('mAP Progression at Key Epochs')
        ax.set_xticks(x[:max(len(baseline_milestones), len(enhanced_milestones))])
        ax.set_xticklabels([f'Ep {ep}' for ep in milestone_epochs[:max(len(baseline_milestones), len(enhanced_milestones))]])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_plot, dpi=100, bbox_inches='tight')
        print(f"✅ Plot saved to: {save_plot}")
        plt.close()
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    """
    Usage:
        python compare_training_results.py
    
    This will compare results from:
    - yolo11m_cbam_sahi_1024px (baseline 85% mAP)
    - yolo11m_ce_fpn_cbam_1024px (target 88-90% mAP)
    """
    
    baseline_path = '/content/drive/MyDrive/kaggle_results/cbam_experiments/yolo11m_cbam_sahi_1024px'
    enhanced_path = '/content/drive/MyDrive/kaggle_results/cbam_experiments/yolo11m_ce_fpn_cbam_1024px'
    
    compare_training_runs(
        baseline_dir=baseline_path,
        enhanced_dir=enhanced_path,
        save_plot='/tmp/ce_fpn_vs_baseline_comparison.png'
    )
