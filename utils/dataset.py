"""
ELEC 475: Lab 2 - Unified Dataset & DataLoader Factory
Supports CIFAR-10 (primary) and Tiny ImageNet-200 (secondary benchmark).
"""
import os
import urllib.request
import zipfile
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10, ImageFolder
from torchvision.datasets.folder import default_loader
from .transforms import get_train_transforms, get_test_transforms

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

TINY_IMAGENET_URLS = [
    "http://cs231n.stanford.edu/tiny-imagenet-200.zip",  # original (often offline)
    # Mirror of the same files, without the top-level 'tiny-imagenet-200/' folder
    "https://huggingface.co/datasets/torch-uncertainty/tiny-imagenet-200/resolve/main/tin.zip",
]

def download_and_extract_tiny_imagenet(data_dir: str = "./data"):
    target_dir = os.path.join(data_dir, "tiny-imagenet-200")
    if os.path.exists(os.path.join(target_dir, "wnids.txt")):
        return target_dir

    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "tiny-imagenet-200.zip")

    if not os.path.exists(zip_path):
        for url in TINY_IMAGENET_URLS:
            try:
                print(f"Downloading Tiny ImageNet-200 from {url}...")
                urllib.request.urlretrieve(url, zip_path)
                print("Download complete.")
                break
            except Exception as e:
                print(f"  Download failed ({e}); trying next source.")
                if os.path.exists(zip_path):
                    os.remove(zip_path)
        else:
            raise RuntimeError("Could not download Tiny ImageNet-200 from any source.")

    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        has_root = all(n.startswith("tiny-imagenet-200/") for n in zip_ref.namelist())
        zip_ref.extractall(data_dir if has_root else target_dir)
    print("Extraction complete.")
    return target_dir


class TinyImageNetVal(Dataset):
    """
    Official Tiny ImageNet-200 validation split: 10,000 images, 50 per class.
    The images sit in one flat folder (val/images) and their labels are listed in val/val_annotations.txt,
    so ImageFolder cannot read it. Labels use the training set's class_to_idx so indices match.
    """
    def __init__(self, root: str, class_to_idx: dict, transform=None):
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.classes = sorted(class_to_idx, key=class_to_idx.get)
        self.samples = []
        with open(os.path.join(root, "val", "val_annotations.txt")) as f:
            for line in f:
                fname, wnid = line.rstrip("\n").split("\t")[:2]
                self.samples.append((os.path.join(root, "val", "images", fname), class_to_idx[wnid]))
        self.targets = [t for _, t in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = default_loader(path)  # converts the few grayscale images to RGB
        if self.transform is not None:
            img = self.transform(img)
        return img, target

def get_dataloaders(
    dataset: str = "cifar10",
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
    pin_memory: bool = True,
    use_augmentation: bool = True,
    target_size: int = 64
):
    dataset_name = dataset.lower()

    if dataset_name == "cifar10":
        train_tf = get_train_transforms("cifar10", augment=use_augmentation, target_size=target_size)
        val_tf   = get_test_transforms("cifar10", target_size=target_size)

        train_set = CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
        val_set   = CIFAR10(root=data_dir, train=False, download=True, transform=val_tf)
        num_classes = 10
        classes = CIFAR10_CLASSES

    elif dataset_name in ["tiny_imagenet", "tinyimagenet"]:
        tiny_dir = download_and_extract_tiny_imagenet(data_dir)
        train_dir = os.path.join(tiny_dir, "train")

        train_tf = get_train_transforms("tiny_imagenet", augment=use_augmentation, target_size=target_size)
        val_tf   = get_test_transforms("tiny_imagenet", target_size=target_size)

        train_set = ImageFolder(root=train_dir, transform=train_tf)
        # Held-out official validation images (the template validated on the training folder)
        val_set   = TinyImageNetVal(tiny_dir, class_to_idx=train_set.class_to_idx, transform=val_tf)
        num_classes = 200
        classes = train_set.classes

    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from 'cifar10', 'tiny_imagenet'.")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=False
    )

    return train_loader, val_loader, num_classes, classes


def get_clean_train_loader(
    dataset: str = "cifar10",
    data_dir: str = "./data",
    batch_size: int = 128,
    num_samples: int = 10000,
    seed: int = 42,
    num_workers: int = 2,
    target_size: int = 64
):
    """
    Fixed random subset of the TRAINING images with the deterministic test transforms (no augmentation).
    Evaluated in eval mode (dropout off, BN running stats), it measures how well the model fits its own
    training data under the same conditions as validation, so Gap = train acc - val acc compares like with like.
    """
    dataset_name = dataset.lower()
    if dataset_name == "cifar10":
        train_set = CIFAR10(root=data_dir, train=True, download=True,
                            transform=get_test_transforms("cifar10", target_size=target_size))
    elif dataset_name in ["tiny_imagenet", "tinyimagenet"]:
        train_dir = os.path.join(download_and_extract_tiny_imagenet(data_dir), "train")
        train_set = ImageFolder(root=train_dir, transform=get_test_transforms("tiny_imagenet", target_size=target_size))
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from 'cifar10', 'tiny_imagenet'.")

    indices = torch.randperm(len(train_set), generator=torch.Generator().manual_seed(seed))
    indices = indices[:min(num_samples, len(train_set))].tolist()
    # Own generator: iterating this loader must not consume the global RNG that drives training shuffles
    return DataLoader(Subset(train_set, indices), batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, generator=torch.Generator().manual_seed(seed))
