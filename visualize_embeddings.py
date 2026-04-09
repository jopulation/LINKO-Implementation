#!/usr/bin/env python3
"""
Visualize LLM embeddings using PCA
"""
import numpy as np
import matplotlib.pyplot as plt
import os

def pca_2d(X):
    """Simple PCA using SVD"""
    X_centered = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    n_components = min(2, U.shape[1])
    return U[:, :n_components], S[:n_components]

def visualize_embeddings():
    base_path = 'saved_files/gpt_code_emb/tx-emb-3-small/include_all_parents2'
    
    # 모든 임베딩 파일 수집
    embedding_files = {
        'DX1 (진단 L1)': 'dx1_gpt_emb.npy',
        'DX2 (진단 L2)': 'dx2_gpt_emb.npy',
        'DX3 (진단 L3)': 'dx3_gpt_emb.npy',
        'RX1 (약물 L1)': 'rx1_gpt_emb.npy',
        'RX2 (약물 L2)': 'rx2_gpt_emb.npy',
        'RX3 (약물 L3)': 'rx3_gpt_emb.npy',
        'PX1 (시술 L1)': 'px1_gpt_emb.npy',
        'PX2 (시술 L2)': 'px2_gpt_emb.npy',
        'PX3 (시술 L3)': 'px3_gpt_emb.npy',
    }
    
    # 3x3 subplot 생성
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    fig.suptitle('LLM Embeddings Visualization (PCA 2D Projection)', fontsize=16, fontweight='bold')
    
    print("🔍 Visualizing embeddings...\n")
    
    for idx, (label, fname) in enumerate(embedding_files.items()):
        path = os.path.join(base_path, fname)
        
        if not os.path.exists(path):
            print(f"⚠️  {label}: 파일 없음")
            continue
        
        # 임베딩 로드
        arr = np.load(path)
        
        # PCA 2D 축소 (SVD 사용)
        reduced, var = pca_2d(arr)
        var_ratio = var / var.sum()
        
        # 작은 데이터셋은 1D인 경우가 있으므로 처리
        if reduced.shape[1] == 1:
            x = reduced[:, 0]
            y = np.zeros_like(x) + np.random.randn(len(x)) * 0.01  # 약간의 노이즈 추가
        else:
            x = reduced[:, 0]
            y = reduced[:, 1]
        
        # 서브플롯에 플롯
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # 스캐터 플롯
        scatter = ax.scatter(x, y, 
                            alpha=0.6, s=40, c=np.arange(arr.shape[0]), 
                            cmap='viridis', edgecolors='black', linewidth=0.3)
        
        # 제목과 라벨
        total_var = f"{var_ratio.sum():.1%}"
        ax.set_title(f"{label}\n({arr.shape[0]} codes, dim={arr.shape[1]})", fontweight='bold')
        ax.set_xlabel(f'PC1 ({var_ratio[0]:.1%})', fontsize=9)
        if reduced.shape[1] > 1:
            ax.set_ylabel(f'PC2 ({var_ratio[1]:.1%})', fontsize=9)
        else:
            ax.set_ylabel('Jitter (noise)', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        print(f"✓ {label:20} | Shape: {str(arr.shape):15} | Var: {total_var}")
    
    plt.tight_layout()
    
    # 저장 및 표시
    output_path = 'results_prompting/embeddings_visualization.png'
    os.makedirs('results_prompting', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 저장됨: {output_path}")
    
    plt.show()
    print("\n✅ 시각화 완료!")

if __name__ == '__main__':
    visualize_embeddings()
