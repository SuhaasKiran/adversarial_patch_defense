import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm
import numpy as np
from data_utils import NORM_MEAN, NORM_STD

TENSOR_MEANS = torch.FloatTensor(NORM_MEAN)[:, None, None]
TENSOR_STD = torch.FloatTensor(NORM_STD)[:, None, None]

def place_patch(img, patch, h_offset=-1, w_offset=-1):
    """Place patch on images at specified or random locations."""
    for i in range(img.shape[0]):
        if h_offset == -1 and w_offset == -1:
            h_offset = np.random.randint(0, img.shape[2] - patch.shape[1] - 1)
            w_offset = np.random.randint(0, img.shape[3] - patch.shape[2] - 1)
        img[i, :, h_offset:h_offset+patch.shape[1], w_offset:w_offset+patch.shape[2]] = patch_forward(patch)
    return img

def patch_forward(patch):
    """Map patch values to ImageNet range."""
    return (torch.tanh(patch) + 1 - 2 * TENSOR_MEANS) / (2 * TENSOR_STD)

def evaluate_patch(model, patch, val_loader, target_class, device='cuda:0'):
    """Evaluate patch effectiveness."""
    model.eval()
    tp, tp_5, counter = 0.0, 0.0, 0.0
    with torch.no_grad():
        for img, img_labels in tqdm(val_loader, desc="Validating...", leave=False):
            for _ in range(4):
                patch_img = place_patch(img, patch)
                patch_img, img_labels = patch_img.to(device), img_labels.to(device)
                pred = model(patch_img)
                tp += torch.logical_and(pred.argmax(dim=-1) == target_class, img_labels != target_class).sum().float()
                tp_5 += torch.logical_and((pred.topk(5, dim=-1)[1] == target_class).any(dim=-1), img_labels != target_class).sum().float()
                counter += (img_labels != target_class).sum()
    acc = tp / counter if counter > 0 else 0.0
    top5 = tp_5 / counter if counter > 0 else 0.0
    return acc, top5

def patch_attack(model, target_class, patch_size=64, num_epochs=5, dataset=None, device='cuda:0'):
    """Train an adversarial patch."""
    train_set, val_set = torch.utils.data.random_split(dataset, [4500, 500])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, drop_last=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=False, drop_last=False, num_workers=4)

    if not isinstance(patch_size, tuple):
        patch_size = (patch_size, patch_size)
    patch = nn.Parameter(torch.zeros(3, patch_size[0], patch_size[1]), requires_grad=True).to(device)
    optimizer = optim.SGD([patch], lr=1e-1, momentum=0.8)
    loss_module = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        t = tqdm(train_loader, leave=False)
        for img, _ in t:
            img = place_patch(img, patch).to(device)
            pred = model(img)
            labels = torch.zeros(img.shape[0], device=pred.device, dtype=torch.long).fill_(target_class)
            loss = loss_module(pred, labels)
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
            t.set_description(f"Epoch {epoch}, Loss: {loss.item():.2f}")

    acc, top5 = evaluate_patch(model, patch, val_loader, target_class, device)
    return patch.data, {"acc": acc.item(), "top5": top5.item()}