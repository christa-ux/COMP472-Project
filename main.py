import torch
import numpy as np
from task2.load import load_data
from task2.extraction import load_resnet18, extract_features
from task2.reduceFeatureDimensions import apply_pca, save_features
from task3.naive_bayes_np import GaussianNaiveBayes
from task3.naive_bayes_sklearn import train_and_predict_sklearn
from task3.evaluation_metrics import evaluate_model, plot_confusion_matrix

def main():
    # Step 1: Check device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Step 2: Load Data
    train_loader, test_loader = load_data(samples_per_class=500, batch_size=32)

    # Step 3: Load Pre-trained ResNet-18
    resnet18 = load_resnet18()
    resnet18.to(device)  # Move the model to the selected device

    # Step 4: Extract Features
    print("Extracting features...")
    train_features, train_labels = extract_features(resnet18, train_loader, device)
    test_features, test_labels = extract_features(resnet18, test_loader, device)

    print(f"Train Features Shape: {train_features.shape}")  # Should be (5000, 512)
    print(f"Test Features Shape: {test_features.shape}")    # Should be (1000, 512)

    # Step 5: Convert tensors to NumPy arrays for PCA
    train_features = train_features.cpu().numpy()  # Move to CPU, convert to NumPy
    test_features = test_features.cpu().numpy()

    # Step 6: Apply PCA to Reduce Dimensions
    print("Applying PCA...")
    train_features_pca, test_features_pca = apply_pca(train_features, test_features, n_components=50)

    print(f"Train Features after PCA: {train_features_pca.shape}")  # (5000, 50)
    print(f"Test Features after PCA: {test_features_pca.shape}")    # (1000, 50)

    # Step 7: Save Features and Labels
    print("Saving features...")
    save_features(train_features_pca, train_labels, test_features_pca, test_labels)

    print("Feature extraction and PCA completed successfully.")

    # Step 8: Train and Evaluate Custom Gaussian Naive Bayes
    print("Training Custom Gaussian Naive Bayes...")
    gnb_custom = GaussianNaiveBayes()
    gnb_custom.fit(train_features_pca, train_labels)  # Train the custom model
    custom_predictions = gnb_custom.predict(test_features_pca)  # Predict on test data

    # Evaluate custom GNB model
    evaluate_model(test_labels, custom_predictions, "Custom GNB")
    plot_confusion_matrix(test_labels, custom_predictions, list(range(10)), "Custom GNB")

    # Step 9: Train and Evaluate Scikit-learn Gaussian Naive Bayes
    print("Training Scikit-learn Gaussian Naive Bayes...")
    sklearn_predictions = train_and_predict_sklearn(train_features_pca, train_labels, test_features_pca)

    # Evaluate sklearn GNB model
    evaluate_model(test_labels, sklearn_predictions, "Scikit-learn GNB")
    plot_confusion_matrix(test_labels, sklearn_predictions, list(range(10)), "Scikit-learn GNB")

if __name__ == "__main__":
    main()
