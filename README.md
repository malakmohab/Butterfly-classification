# Butterfly-classification
🦋 Butterfly Species Classifier
This project implements a machine learning pipeline for butterfly species classification using feature-based image analysis. It applies Histogram of Oriented Gradients (HOG) for feature extraction, PCA and LDA for dimensionality reduction, and SVM for classification. The solution is designed for fast and scalable training on labeled butterfly image datasets.

📂 Dataset
You will need a dataset of butterfly images, organized as follows:

graphql
Copy
Edit
Butterflies/
├── train/                     # Folder containing training images
├── test/                      # Folder containing test images
├── Training_set.csv           # CSV with columns: filename, label
├── Testing_set.csv            # CSV with column: filename
Note: You must replace the paths in the script (e.g., /Users/habib/Desktop/Arabic Sign Language/Butterflies/...) with your own file paths depending on where you store the dataset locally.
Dataset:https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification

🚀 Features & Workflow
✅ Step 1: Preprocessing
Resizes images to 64×64 for computational efficiency

Converts to grayscale

Extracts HOG features to capture shape and texture

✅ Step 2: Feature Engineering
StandardScaler to normalize data

PCA to retain 95% variance

LDA for supervised class separation (optional for visualization)

✅ Step 3: Classification
SVM (RBF kernel) is trained on PCA-transformed features

Confusion Matrix to visualize model performance

Classification Report with accuracy, precision, recall, and F1-score

📦 Dependencies
Make sure the following Python packages are installed:
pip install numpy pandas matplotlib pillow scikit-learn seaborn tqdm scikit-image joblib
🧠 How to Use
Download or prepare a labeled butterfly image dataset

Place images and CSVs in a folder (e.g., Butterflies/)

CSVs should match image filenames exactly

Adjust Paths in the Script

Replace all hardcoded paths (like "/Users/habib/Desktop/...") with your own local paths

Run the Script
python butterfly_classifier.py
📊 Visualizations
📌 PCA Scatter Plot: Visualizes the data's variance in 2D

📌 LDA Scatter Plot: Shows class separation if multiple components

📌 Explained Variance Curve: Shows how many PCA components preserve 95% variance

📌 Confusion Matrix: Helps evaluate model accuracy per class

📝 Outputs
Trained SVM classifier

HOG feature matrix (in memory)

PCA and LDA transformed data

Visualization of projections and metrics

Printed classification report with detailed stats
