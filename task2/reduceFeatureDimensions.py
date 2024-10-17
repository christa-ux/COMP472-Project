import numpy as np
from sklearn.decomposition import PCA

def apply_pca(train_features, test_features, n_components=50):
    """Apply PCA to reduce feature dimensions."""
    pca = PCA(n_components=n_components)
    train_features_pca = pca.fit_transform(train_features)  # Fit on train data
    test_features_pca = pca.transform(test_features)  # Transform test data
    return train_features_pca, test_features_pca

# this is optional but i think its a good idea cuz we might need this later and this is a good way of caching featyres
def save_features(train_features, train_labels, test_features, test_labels):
    """Save features and labels to .npy files."""
    # why npy???
    # saving files in .npy format when working with NumPy arrays because it provides several advantages for scientific computing.
    np.save('./Saved-features/train_features_pca.npy', train_features)
    np.save('./Saved-features/train_labels.npy', train_labels.numpy())
    np.save('./Saved-features/test_features_pca.npy', test_features)
    np.save('./Saved-features/test_labels.npy', test_labels.numpy())
