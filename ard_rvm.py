import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.linalg import inv

class ARDRVM:
    def __init__(self, alpha=1e-6, beta=1.0, max_iter=100, tol=1e-3):
        self.alpha_init = alpha   # initial precision for all weights
        self.beta = beta          # initial noise precision
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = None
        self.mean = None
        self.cov = None
        self.relevance = None   # indices of kept features

    def fit(self, X, y):
        """
        Fit RVM with ARD.
        X: (n_samples, n_features)
        y: (n_samples,)
        """
        n, d = X.shape
        alpha = np.ones(d) * self.alpha_init
        beta = self.beta
        # Compute sufficient statistics: S = beta * X.T @ X + diag(alpha)
        # and mean = beta * S^{-1} X.T y
        # Use iterative algorithm (Tipping & Faul)
        # We'll use the original fast method: start with one basis function and add/delete.
        # For simplicity, we use the classic EM algorithm.

        # Classic EM for RVM:
        # E-step: compute posterior covariance and mean given current alpha, beta.
        # M-step: update alpha_i = 1 / (mean_i^2 + cov_ii), beta = (n - sum_i (1 - alpha_i * cov_ii)) / ||y - X mean||^2
        # This is simpler and works well for moderate d.

        for it in range(self.max_iter):
            # Compute posterior covariance and mean
            XtX = X.T @ X
            # S = beta * XtX + diag(alpha)
            S = beta * XtX + np.diag(alpha)
            # Cholesky for stability
            try:
                L = cholesky(S, lower=True)
                # Solve for mean: S mean = beta X^T y
                Xt_y = X.T @ y
                mean = solve_triangular(L, beta * Xt_y, lower=True)
                mean = solve_triangular(L.T, mean, lower=False)
                # Compute diagonal of covariance (for variance of weights)
                # We need inv(S) diagonal. Solve for each unit vector.
                invS_diag = np.zeros(d)
                for i in range(d):
                    e = np.zeros(d)
                    e[i] = 1.0
                    v = solve_triangular(L, e, lower=True)
                    v = solve_triangular(L.T, v, lower=False)
                    invS_diag[i] = v[i]
            except np.linalg.LinAlgError:
                # If singular, use pseudo-inverse
                invS = np.linalg.pinv(S)
                mean = beta * invS @ Xt_y
                invS_diag = np.diag(invS)
            # Update alpha
            alpha_new = 1.0 / (mean**2 + invS_diag)
            # Update beta
            residuals = y - X @ mean
            err2 = np.sum(residuals**2)
            gamma = 1.0 - alpha * invS_diag
            beta_new = (n - np.sum(gamma)) / err2
            # Check convergence
            if np.max(np.abs(alpha_new - alpha)) < self.tol and abs(beta_new - beta) < self.tol:
                alpha = alpha_new
                beta = beta_new
                break
            alpha = alpha_new
            beta = beta_new
        # Keep only relevant features (alpha < 1e6)
        relevant = alpha < 1e6
        self.relevance = np.where(relevant)[0]
        self.alpha = alpha
        self.beta = beta
        self.mean = mean
        self.cov_inv_diag = invS_diag
        # Store pruned model
        if len(self.relevance) > 0:
            self.reduced_mean = mean[relevant]
            self.reduced_X = X[:, relevant]
        else:
            self.reduced_mean = np.array([])
            self.reduced_X = np.empty((n, 0))

    def predict(self, X):
        """Predict using the relevance vectors (pruned features)."""
        if len(self.relevance) == 0:
            return np.zeros(X.shape[0])
        # Use only relevant features
        X_rel = X[:, self.relevance]
        return X_rel @ self.reduced_mean

    def get_relevance_counts(self):
        return len(self.relevance)
