import torch
import torchvision
from tqdm.notebook import tqdm

def load_pretrained_model(device, checkpoint_path):
    """Load pretrained ResNet34 model."""
    os.environ["TORCH_HOME"] = checkpoint_path
    model = torchvision.models.resnet34(weights='IMAGENET1K_V1').to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

def evaluate_model(model, data_loader, img_func=None, device='cuda:0'):
    """Evaluate model accuracy on dataset."""
    model.eval()
    tp, tp_5, counter = 0.0, 0.0, 0.0
    for imgs, labels in tqdm(data_loader, desc="Validating..."):
        imgs, labels = imgs.to(device), labels.to(device)
        if img_func is not None:
            imgs = img_func(imgs, labels)
        with torch.no_grad():
            preds = model(imgs)
        tp += (preds.argmax(dim=-1) == labels).sum().float()
        tp_5 += (preds.topk(5, dim=-1)[1] == labels[..., None]).any(dim=-1).sum().float()
        counter += preds.shape[0]
    acc = tp / counter
    top5 = tp_5 / counter
    print(f"Top-1 accuracy: {acc*100:.2f}%")
    print(f"Top-5 accuracy: {top5*100:.2f}%")
    return acc.item(), top5.item()