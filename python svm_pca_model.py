
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_images_from_directory(directory, img_size=(128, 128)):
    X = []
    y = []
    for label in ['First Print', 'Second Print']:
        class_dir = os.path.join(directory, label)
        for img_file in os.listdir(class_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, img_file)
                img = load_img(img_path, color_mode='grayscale', target_size=img_size)
                img_array = img_to_array(img).flatten()
                X.append(img_array)
                y.append(label)
    return np.array(X), y

def plot_pca_variance(pca_model):
    explained_var = pca_model.explained_variance_ratio_
    cum_var = np.cumsum(explained_var)

    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_var) + 1), explained_var, alpha=0.5, label='Individual Explained Variance')
    plt.plot(range(1, len(cum_var) + 1), cum_var, marker='o', color='r', label='Cumulative Explained Variance')
    plt.axhline(y=0.99, color='g', linestyle='--', label='99% variance threshold')
    plt.xlabel('Principal Component Index')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA Explained Variance')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def apply_pca(X_train, X_test, n_components=100):
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca, pca

def train_svm_classifier(X_train_pca, y_train):
    svm = SVC(kernel='rbf', probability=True, tol=1e-6)
    svm.fit(X_train_pca, y_train)
    return svm

def evaluate_model(model, X_test_pca, y_test):
    y_pred = model.predict(X_test_pca)
    y_proba = model.predict_proba(X_test_pca)[:, 1]

    print("Test Accuracy:", accuracy_score(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['First Print', 'Second Print']))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['First Print', 'Second Print'])
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()

    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'SVM ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    dataset_dir = "/content/dataset"

    print("\nLoading images...")
    X, y = load_images_from_directory(dataset_dir)
    y = LabelEncoder().fit_transform(y)

    print("\nSplitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\nApplying PCA...")
    X_train_pca, X_test_pca, pca_model = apply_pca(X_train, X_test, n_components=100)

    print("\nVisualizing PCA variance...")
    plot_pca_variance(pca_model)

    print("\nTraining SVM classifier...")
    model = train_svm_classifier(X_train_pca, y_train)

    print("\nEvaluating model...")
    evaluate_model(model, X_test_pca, y_test)

if __name__ == "__main__":
    main()
