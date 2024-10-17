
# CIFAR-10 Image Classification

## Project Overview
This project demonstrates the use of **feature extraction** with a pre-trained **ResNet-18 model** and **dimensionality reduction using PCA** for classifying CIFAR-10 images. We further implement **Gaussian Naive Bayes** classifiers using both a custom implementation and Scikit-learn’s built-in version to evaluate the classification performance.

---

## **Installation and Setup**

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd project/
   
2. **Set up a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/Mac
    venv\Scripts\activate     # On Windows

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
4. **Ensure PyTorch detects CUDA (if you plan to use GPU, which would reduce overhead since we are image processing):**
    ```bash
    python -c "import torch; print(torch.cuda.  is_available())"

- If `False` is returned, verify your PyTorch installation or CUDA setup.

---
## **How to Run the Project**

### Run the main.py script:

    
        python main.py
### Program Output:

- The program will print feature shapes, apply PCA, and save the reduced features.
- It will train both a custom Gaussian Naive Bayes and Scikit-learn’s GNB on the CIFAR-10 data.
- Evaluation metrics (Accuracy, Precision, Recall, and F1-Score) will be printed.
- Confusion matrices will be generated for both classifiers and saved to the working directory.

**MORE WILL BE ADDED**

---

## **Description of Tasks**

### Feature Extraction (Task 2)

We use a pre-trained ResNet-18 model (with the last fully connected layer removed) to extract 512-dimensional feature vectors from the CIFAR-10 images.
The images are resized to 224x224 and normalized to match the ImageNet pre-processing standards.
Dimensionality Reduction with PCA

After feature extraction, PCA reduces the feature vectors from 512 to 50 dimensions to improve computational efficiency for the Naive Bayes classifiers.

### **Naive Bayes Classifiers (Task 3)**

We implement two versions of Gaussian Naive Bayes:
Custom implementation using basic Python and NumPy.
Scikit-learn’s GaussianNB for comparison.
Both classifiers are trained on the PCA-reduced features.
Model Evaluation

Accuracy, Precision, Recall, and F1-Score metrics are computed for both models.
Confusion matrices are generated to visualize the performance across the 10 CIFAR-10 classes.
Evaluation Metrics
The following metrics are used to evaluate the models:

- Accuracy: Proportion of correct predictions over the total number of samples.
- Precision: Number of true positive predictions divided by all positive predictions.
- Recall: Number of true positive predictions divided by all actual positives.
- F1-Score: Harmonic mean of Precision and Recall.
- A sample confusion matrix is generated for each model to visualize classification performance across the 10 CIFAR-10 classes.

---
## Dependencies

- Python 3.x  
- PyTorch  
- NumPy  
- Scikit-learn  
- Matplotlib (for plotting confusion matrices)  

### Install Dependencies

Use the following command to install the required packages: 

    pip install torch torchvision torchaudio numpy scikit-learn matplotlib

---

## **Known Issues and Troubleshooting**
### CUDA Not Detected:

Ensure you installed the correct version of PyTorch with CUDA support. Try:
    
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118




### PCA Not Reducing Dimensions:

Ensure that the extracted features are converted to NumPy arrays before applying PCA:

    train_features = train_features.cpu().numpy()
    test_features = test_features.cpu().numpy()

---

## **Acknowledgments**
- CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
- PyTorch Transfer Learning Tutorial: https://pytorch.org/tutorials/
- Scikit-learn Documentation: https://scikit-learn.org/


