import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE


def plot_tsne_comparison_grid(cvm_dir, acvm_dir, tasks=[0, 4, 9], save_path="tsne_comparison.png"):

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    methods = [
        {"name": "CVM (Baseline)", "dir": cvm_dir, "row": 0},
        {"name": "ACVM (Ours)", "dir": acvm_dir, "row": 1}
    ]

    tsne = TSNE(n_components=2, perplexity=35, random_state=1234, init='pca', learning_rate='auto')

    for method in methods:
        for col, task_id in enumerate(tasks):
            ax = axes[method["row"], col]

            file_path = os.path.join(method["dir"], f"features_task_{task_id}.npz")
            if not os.path.exists(file_path):
                ax.text(0.5, 0.5, f"Missing Data\nTask {task_id}", ha='center', va='center', fontsize=14)
                ax.axis('off')
                continue

            print(f"[*] Processing {method['name']} - Task {task_id}...")
            data = np.load(file_path)
            embs = data['embs']
            labels = data['labels']
            anchors = data['anchors']
            seen_indices = data['seen_indices']

            combined_data = np.vstack([embs, anchors])
            tsne_results = tsne.fit_transform(combined_data)

            n_samples = embs.shape[0]
            vis_embs = tsne_results[:n_samples]
            vis_anchors = tsne_results[n_samples:]

            palette = sns.color_palette("tab20", n_colors=max(seen_indices) + 1)

            for idx, cls in enumerate(seen_indices):
                cls_mask = labels == cls

                ax.scatter(vis_embs[cls_mask, 0], vis_embs[cls_mask, 1],
                           color=palette[cls], alpha=0.3, s=10)

                ax.scatter(vis_anchors[idx, 0], vis_anchors[idx, 1],
                           color=palette[cls], marker='*', s=350, edgecolors='black', linewidths=1.2)

            ax.set_xticks([])
            ax.set_yticks([])

            if method["row"] == 0:
                ax.set_title(f"After Task {task_id}", fontsize=18, fontweight='bold', pad=15)

            if col == 0:
                ax.set_ylabel(method['name'], fontsize=18, fontweight='bold', labelpad=15)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.1)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[+] Successfully saved comparison grid to: {save_path}")


if __name__ == "__main__":
    CVM_FEATURES_DIR = "checkpoints/tsne_data_CVM"
    ACVM_FEATURES_DIR = "checkpoints/tsne_data_ACVM"

    plot_tsne_comparison_grid(
        cvm_dir=CVM_FEATURES_DIR,
        acvm_dir=ACVM_FEATURES_DIR,
        tasks=[0, 4, 9],
        save_path="results/tsne_comparison_grid.png"
    )