import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0] = 1.0

    def transform(self, X):
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class GaussianNaiveBayes:
    def __init__(self):
        self.classes_ = None
        self.priors_ = None
        self.means_ = None
        self.vars_ = None
        self.epsilon_ = 1e-9

    def fit(self, X, y):
        self.classes_, class_counts = np.unique(y, return_counts=True)
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape

        self.priors_ = class_counts / n_samples
        self.means_ = np.zeros((n_classes, n_features), dtype=np.float64)
        self.vars_ = np.zeros((n_classes, n_features), dtype=np.float64)

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.means_[idx, :] = np.mean(X_c, axis=0)
            self.vars_[idx, :] = np.var(X_c, axis=0) + self.epsilon_

    def _pdf(self, class_idx, x):
        mean = self.means_[class_idx]
        var = self.vars_[class_idx]
        
        numerator = np.exp(- (x - mean)**2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def _log_likelihood(self, class_idx, x):
        mean = self.means_[class_idx]
        var = self.vars_[class_idx]
        log_prob = -0.5 * np.sum(np.log(2 * np.pi * var) + ((x - mean)**2 / var), axis=1)
        return log_prob

    def predict_proba(self, X):
        log_probs = np.zeros((X.shape[0], len(self.classes_)))

        for idx, c in enumerate(self.classes_):
            prior_log = np.log(self.priors_[idx])
            likelihood_log = self._log_likelihood(idx, X)
            log_probs[:, idx] = prior_log + likelihood_log

        max_log_probs = np.max(log_probs, axis=1, keepdims=True)
        exp_probs = np.exp(log_probs - max_log_probs)
        return exp_probs / np.sum(exp_probs, axis=1, keepdims=True)

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train_ = None
        self.y_train_ = None
        self.classes_ = None

    def fit(self, X, y):
        self.X_train_ = np.array(X)
        self.y_train_ = np.array(y)
        self.classes_ = np.unique(y)

    def _compute_distances(self, X):
        n_test = X.shape[0]
        n_train = self.X_train_.shape[0]
        dists = np.zeros((n_test, n_train))
        
        for i in range(n_test):
            diff = self.X_train_ - X[i]
            dists[i, :] = np.sqrt(np.sum(diff**2, axis=1))
            
        return dists

    def predict_proba(self, X):
        distances = self._compute_distances(X)
        n_test = X.shape[0]
        probs = np.zeros((n_test, len(self.classes_)))
        
        for i in range(n_test):
            k_indices = np.argsort(distances[i])[:self.k]
            k_nearest_labels = self.y_train_[k_indices]
            
            for idx, c in enumerate(self.classes_):
                probs[i, idx] = np.sum(k_nearest_labels == c) / self.k
                
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
