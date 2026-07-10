import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import argparse


def plot_tsne_grid(cvm_dir=None, acvm_dir=None, tasks=[0, 4, 9], save_path="tsne_grid.png"):
    methods = []
    if cvm_dir:
        methods.append({"name": "CVM (Baseline)", "dir": cvm_dir})
    if acvm_dir:
        methods.append({"name": "ACVM (Ours)", "dir": acvm_dir})

    if not methods:
        print("[-] Error: You must provide at least one directory (--cvm_dir or --acvm_dir)!")
        return

    num_rows = len(methods)
    fig, axes = plt.subplots(num_rows, len(tasks), figsize=(18, 6 * num_rows))

    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    tsne = TSNE(n_components=2, perplexity=35, random_state=1234, init='pca', learning_rate='auto')

    for row, method in enumerate(methods):
        for col, task_id in enumerate(tasks):
            ax = axes[row, col]

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

            if row == 0:
                ax.set_title(f"After Task {task_id}", fontsize=18, fontweight='bold', pad=15)
            if col == 0:
                ax.set_ylabel(method['name'], fontsize=18, fontweight='bold', labelpad=15)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.1)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[+] Successfully saved grid to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot t-SNE grid flexibly")
    parser.add_argument('--cvm_dir', type=str, default=None, help="Path to CVM baseline features")
    parser.add_argument('--acvm_dir', type=str, default=None, help="Path to ACVM features")
    parser.add_argument('--out', type=str, default="tsne_grid.png", help="Output image path")

    args = parser.parse_args()

    plot_tsne_grid(
        cvm_dir=args.cvm_dir,
        acvm_dir=args.acvm_dir,
        tasks=[0, 4, 9],
        save_path=args.out
    )