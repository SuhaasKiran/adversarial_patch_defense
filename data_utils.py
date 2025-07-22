import os
import json
import urllib.request
from urllib.error import HTTPError
import zipfile
import torch
import torchvision
from torchvision import transforms
import pytorch_lightning as pl

# Constants for dataset paths
DATASET_PATH = "../data"
CHECKPOINT_PATH = "../saved_models/"

# ImageNet normalization constants
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

def download_and_extract(base_url, pretrained_files):
    """Download and extract dataset and pretrained model files."""
    os.makedirs(DATASET_PATH, exist_ok=True)
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    for dir_name, file_name in pretrained_files:
        file_path = os.path.join(dir_name, file_name)
        if not os.path.isfile(file_path):
            file_url = base_url + file_name
            print(f"Downloading {file_url}...")
            try:
                urllib.request.urlretrieve(file_url, file_path)
            except HTTPError as e:
                print(f"HTTP Error: {e}")
            if file_name.endswith(".zip"):
                print(f"Unzipping {file_name}...")
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(file_path.rsplit("/", 1)[0])

def load_dataset():
    """Load Tiny ImageNet dataset with normalization."""
    pl.seed_everything(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    plain_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    imagenet_path = os.path.join(DATASET_PATH, "TinyImageNet/")
    if not os.path.isdir(imagenet_path):
        raise FileNotFoundError(
            f"Could not find TinyImageNet dataset at {imagenet_path}. "
            f"Please download the dataset or update {DATASET_PATH}."
        )

    dataset = torchvision.datasets.ImageFolder(root=imagenet_path, transform=plain_transforms)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=False, drop_last=False, num_workers=8
    )

    with open(os.path.join(imagenet_path, "label_list.json"), "r") as f:
        label_names = json.load(f)

    return dataset, data_loader, label_names

def get_label_index(label_str, label_names):
    """Get index of a label string in label_names."""
    if label_str not in label_names:
        raise ValueError(f"Label '{label_str}' not found. Check spelling.")
    return label_names.index(label_str)