import torch
import numpy as np
import matplotlib.pyplot as plt
from data_utils import NORM_MEAN, NORM_STD
from tabulate import tabulate
from IPython.display import display, HTML

def show_prediction(img, label, pred, label_names, K=5, adv_img=None):
    """Display image with true label and top-K predictions."""
    if isinstance(img, torch.Tensor):
        img = img.cpu().permute(1, 2, 0).numpy()
        img = (img * np.array(NORM_STD)[None, None]) + np.array(NORM_MEAN)[None, None]
        img = np.clip(img, 0.0, 1.0)
        label = label.item()

    fig = plt.figure(figsize=(12, 2) if adv_img is None else (12, 2), gridspec_kw={'width_ratios': [1, 1] if adv_img is not None else [1]})
    ax = fig.subplots(1, 2 if adv_img is not None else 1)

    if not isinstance(ax, np.ndarray):
        ax = [ax]

    ax[0].imshow(img)
    ax[0].set_title(label_names[label])
    ax[0].axis('off')

    if adv_img is not None:
        adv_img = adv_img.cpu().permute(1, 2, 0).numpy()
        adv_img = (adv_img * np.array(NORM_STD)[None, None]) + np.array(NORM_MEAN)[None, None]
        adv_img = np.clip(adv_img, 0.0, 1.0)
        ax[1].imshow(adv_img)
        ax[1].set_title('Adversarial')
        ax[1].axis('off')

    if abs(pred.sum().item() - 1.0) > 1e-4:
        pred = torch.softmax(pred, dim=-1)
    topk_vals, topk_idx = pred.topk(K, dim=-1)
    topk_vals, topk_idx = topk_vals.cpu().numpy(), topk_idx.cpu().numpy()
    print("Predictions:")
    for i in range(topk_vals.shape[0]):
        print(f"{label_names[topk_idx[i]]}: {topk_vals[i]*100:.4f}%")

    plt.show()
    plt.close()

def show_patches(patch_dict, class_names, patch_sizes):
    """Visualize adversarial patches for different classes and sizes."""
    fig, ax = plt.subplots(len(patch_sizes), len(class_names), figsize=(len(class_names)*2.2, len(patch_sizes)*2.2))
    for c_idx, cname in enumerate(class_names):
        for p_idx, psize in enumerate(patch_sizes):
            patch = patch_dict[cname][psize]["patch"]
            patch = (torch.tanh(patch) + 1) / 2
            patch = patch.cpu().permute(1, 2, 0).numpy()
            patch = np.clip(patch, 0.0, 1.0)
            ax[p_idx][c_idx].imshow(patch)
            ax[p_idx][c_idx].set_title(f"{cname}, size {psize}")
            ax[p_idx][c_idx].axis('off')
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.show()

def show_table(patch_dict, class_names, patch_sizes, top_1=True):
    """Display table of patch performance metrics."""
    i = 0 if top_1 else 1
    table = [
        [name] + [f"{100.0 * patch_dict[name][psize]['results'][i]:.2f}%" for psize in patch_sizes]
        for name in class_names
    ]
    display(HTML(tabulate(
        table, tablefmt='html', headers=["Class name"] + [f"Patch size {psize}x{psize}" for psize in patch_sizes]
    )))