import numpy as np
import argparse
import csv
from classifier import GaussianNaiveBayes, KNN, StandardScaler
from utils import normalize_landmarks, extract_pairwise_distances
from config import ALL_WORDS

def load_dataset(csv_path, filter_words=True):
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        data = list(reader)
        
    if filter_words:
        data = [row for row in data if row[-1] in ALL_WORDS]
    
    data_np = np.array(data)
    X = data_np[:, :-1].astype(float)
    y = data_np[:, -1]
    
    return X, y

def train_test_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    split_idx = int(len(X) * (1 - test_size))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def main():
    parser = argparse.ArgumentParser(description="Train ASL Models")
    parser.add_argument("--method", type=str, choices=['bayes', 'knn', 'rf'], required=True)
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    X, y = load_dataset("asl_hand_landmarks.csv")
    X = normalize_landmarks(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

    if args.method == 'bayes':
        model = GaussianNaiveBayes()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = np.mean(y_pred == y_test) * 100
        print(f"Validation Accuracy: {accuracy:.2f}%")
        
        np.savez('bayes_model.npz', classes=model.classes_, class_prior=model.priors_, 
                 class_mean=model.means_, class_var=model.vars_,
                 scaler_mean=scaler.mean_, scaler_scale=scaler.scale_)
        
    elif args.method == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        import joblib
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = np.mean(y_pred == y_test) * 100
        print(f"Validation Accuracy: {accuracy:.2f}%")
        
        joblib.dump({'model': model, 'scaler_mean': scaler.mean_, 'scaler_scale': scaler.scale_}, 'rf_model.joblib')
        
    elif args.method == 'knn':
        model = KNN(k=args.k)
        model.fit(X_train, y_train)
        
        np.savez(f'knn_model_k{args.k}.npz',
                 X_train=model.X_train_,
                 y_train=model.y_train_,
                 classes=model.classes_,
                 k=np.array([args.k]),
                 scaler_mean=scaler.mean_,
                 scaler_scale=scaler.scale_)

    predictions = model.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
