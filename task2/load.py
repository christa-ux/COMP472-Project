import torchvision
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms


def get_transforms():
    """Returns the transformations required for ResNet-18."""
    return transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224 as required by ResNet-18
        transforms.ToTensor(),  # Convert image to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])


def get_subset(dataset, samples_per_class):
    """Get the first N samples per class from a dataset."""
    class_counts = {i: 0 for i in range(10)}
    indices = []

    for idx, (_, label) in enumerate(dataset):
        if class_counts[label] < samples_per_class:
            indices.append(idx)
            class_counts[label] += 1
        if sum(class_counts.values()) == 10 * samples_per_class:
            break

    return Subset(dataset, indices)


def load_data(samples_per_class=500, batch_size=32):
    """Load CIFAR-10 datasets and create DataLoaders."""
    transform = get_transforms()

    # Load CIFAR-10
    train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Create subsets
    train_subset = get_subset(train_data, samples_per_class)
    test_subset = get_subset(test_data, 100)

    # DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader
