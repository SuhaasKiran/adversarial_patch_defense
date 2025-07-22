import torch
import numpy as np
from patch_detector import PatchDetector
from data_utils import NORM_MEAN, NORM_STD
from model_utils import evaluate_model

def initialize_defense():
    """Initialize patch detectors for defense."""
    black_old = [0.0, 0.0, 0.0]
    black_new = [(b - NORM_MEAN[i]) / NORM_STD[i] for i, b in enumerate(black_old)]
    print(f"Normalized black values: {black_new}")

    patch_size = 48
    A1, A2 = 88, 88
    B1, B2 = A1 + patch_size, A2 + patch_size
    offsets = [A1, A2, B1, B2]

    sac_processor = PatchDetector(3, 1, base_filter=16, square_sizes=[100, 75, 60, 50], n_patch=1)
    sac_processor2 = PatchDetector(
        3, 1, base_filter=64, square_sizes=[75, 60, 50], n_patch=1,
        mask_type=0, iou_test=True, offsets=offsets
    )

    sac_processor.unet.load_state_dict(torch.load("ckpts/coco_at.pth", map_location='cpu'))
    sac_processor2.unet.load_state_dict(torch.load("ckpts/apricot_mask.pth", map_location='cpu'))

    return sac_processor, sac_processor2

def perform_patch_attack_with_defense(model, patch, exmp_batch, label_batch, sac_processor2, device='cuda:0', defend=True, h_offset=154, w_offset=22):
    """Perform patch attack with optional defense."""
    patch_batch = exmp_batch.clone()
    patch_batch = patch_utils.place_patch(patch_batch, patch, h_offset, w_offset)

    images_sac = np.zeros(patch_batch.shape)
    ious = []

    if defend:
        for i in range(patch_batch.shape[0]):
            img = np.expand_dims(patch_batch[i], axis=0)
            x_processed, mask, raw_mask, iou_val = sac_processor2(
                torch.tensor(img), bpda=True, shape_completion=True, simple_shape_completion=False
            )
            images_sac[i] = x_processed[0].cpu().detach().numpy()
            ious.append(iou_val)

    with torch.no_grad():
        patch_preds = model(patch_batch.to(device))
        sac_preds = model(torch.tensor(images_sac, dtype=torch.float32).to(device)) if defend else patch_preds

    acc, top5, patch_acc, patch_top5 = get_accuracy(model, label_batch, patch_preds, sac_preds, device)
    iou_avg = sum(ious) / len(ious) if ious else 0.0
    return acc, top5, patch_acc, patch_top5, iou_avg

def get_accuracy(model, label_batch, patch_preds, sac_preds, device='cuda:0'):
    """Calculate accuracy metrics for patch and defense predictions."""
    model.eval()
    tp, tp_5, tp_patch, tp_5_patch, counter = 0.0, 0.0, 0.0, 0.0, 0.0

    tp += (patch_preds.argmax(dim=-1) == label_batch).sum().float()
    tp_5 += (patch_preds.topk(5, dim=-1)[1] == label_batch[..., None]).any(dim=-1).sum().float()
    counter += patch_preds.shape[0]
    tp_patch += (sac_preds.argmax(dim=-1) == label_batch).sum().float()
    tp_5_patch += (sac_preds.topk(5, dim=-1)[1] == label_batch[..., None]).any(dim=-1).sum().float()

    acc = tp / counter
    top5 = tp_5 / counter
    patch_acc = tp_patch / counter
    patch_top5 = tp_5_patch / counter
    return acc.item(), top5.item(), patch_acc.item(), patch_top5.item()