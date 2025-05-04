import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path

data_path = Path(__file__).parent / 'data'

# Define a custom transform to binarize the image
class Binarize(object):
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, img):
        # Convert to tensor (if not already)
        img = transforms.ToTensor()(img)
        # Binarize using the threshold
        return (img > self.threshold).float()


def build_dataloaders(batch_size: int = 64, shuffle_train=True):
    # Compose transforms: binarize after resizing
    transform = transforms.Compose([
        Binarize(threshold=0.5),
    ])

    # Load MNIST training and test datasets with the binary transform
    train_dataset = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
