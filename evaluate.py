import numpy as np
from classifier import GaussianNaiveBayes, KNN, StandardScaler
from train import load_dataset, train_test_split
from utils import normalize_landmarks

def confusion_matrix(y_true, y_pred, classes):
    n = len(classes)
    matrix = np.zeros((n, n), dtype=int)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for yt, yp in zip(y_true, y_pred):
        if yt in class_to_idx and yp in class_to_idx:
            matrix[class_to_idx[yt], class_to_idx[yp]] += 1
    return matrix

def print_confusion_matrix(matrix, classes):
    print("\nConfusion Matrix:")
    header = "      " + " ".join([f"{str(c):>3}" for c in classes])
    print(header)
    print("   " + "-" * len(header))
    for i, row in enumerate(matrix):
        print(f"{str(classes[i]):>3} | " + "".join([f"{val:>4}" for val in row]))

def evaluate():
    print("Loading test data...")
    X, y = load_dataset("asl_hand_landmarks.csv")
    
    X = normalize_landmarks(X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    print("\n=== Gaussian Naive Bayes ===")
    nb = GaussianNaiveBayes()
    nb.fit(X_train, y_train)
    nb_preds = nb.predict(X_test)
    nb_acc = np.mean(nb_preds == y_test)
    print(f"Accuracy: {nb_acc * 100:.2f}%")
    
    cm_nb = confusion_matrix(y_test, nb_preds, nb.classes_)
    print_confusion_matrix(cm_nb, nb.classes_)

    print("\n=== KNN Analysis ===")
    k_values = [1, 3, 5, 7, 10]
    
    for k in k_values:
        knn = KNN(k=k)
        knn.fit(X_train, y_train)
        
        train_preds = knn.predict(X_train)
        train_acc = np.mean(train_preds == y_train)
        
        test_preds = knn.predict(X_test)
        test_acc = np.mean(test_preds == y_test)
        
        print(f"k={k:2d} | Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%")

if __name__ == "__main__":
    evaluate()
