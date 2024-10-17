import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


def load_resnet18():
    """Load ResNet-18 without the final classification layer."""
    #resnet18 = models.resnet18(pretrained=True)
    # Use weights parameter instead of pretrained=True
    weights = ResNet18_Weights.IMAGENET1K_V1  # You can also use ResNet18_Weights.DEFAULT for latest weights
    resnet18 = models.resnet18(weights=weights)
    resnet18 = nn.Sequential(*list(resnet18.children())[:-1])  # Remove the last layer
    return resnet18

# Generic function to display predictions for a few images
# got the format from pyTorch tut
def extract_features(model, data_loader, device):
    """Extract features from images using the given model."""
    features, labels = [], []

    model.to(device)  # Move model to GPU if available
    model.eval()  # Set to evaluation mode

    with torch.no_grad():  # Disable gradient calculation
        for images, label_batch in data_loader:
            images = images.to(device)  # Move images to the appropriate device
            output = model(images)  # Forward pass
            output = output.view(output.size(0), -1)  # Flatten to (batch_size, 512)
            features.append(output.cpu())  # Move to CPU
            labels.append(label_batch)

    return torch.cat(features), torch.cat(labels)  # Concatenate all features and labels
