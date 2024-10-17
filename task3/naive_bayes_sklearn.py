# Scikit-learn Naive Bayes

from sklearn.naive_bayes import GaussianNB

def train_and_predict_sklearn(train_features, train_labels, test_features):
    """Train and predict using Scikit-learn's Gaussian Naive Bayes."""
    model = GaussianNB()
    model.fit(train_features, train_labels)
    predictions = model.predict(test_features)
    return predictions
