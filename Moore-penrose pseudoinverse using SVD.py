import numpy as np

# (a) Moore-Penrose Pseudoinverse using SVD

def pseudoinverse_svd(A, tol=1e-15):
    
 # Compute the Moore-Penrose pseudoinverse using SVD.

    U, S, VT = np.linalg.svd(A, full_matrices=False)
    
    # Invert singular values above tolerance
    S_plus = np.array([1/s if s > tol else 0 for s in S])
    
    # Construct diagonal matrix Sigma^+
    S_plus = np.diag(S_plus)
    
    # Compute pseudoinverse
    A_plus = VT.T @ S_plus @ U.T
    return A_plus
    


# b) Test on random invertible square matrices

N = 1000
mismatch_count_square = 0
np.random.seed(0)

for _ in range(N):
    A = np.random.randn(5, 5)  
    A_plus = pseudoinverse_svd(A)
    A_inv = np.linalg.inv(A)
    
    if not np.allclose(A_plus, A_inv, atol=1e-10):
        mismatch_count_square += 1

print(f"(b) Number of mismatches in {N} square matrix trials: {mismatch_count_square}")

# (c) Test against ridge regression using SVD (numerically stable)

m, n = 5, 10  # More columns than rows
mismatch_count_rect = 0
lam = 1e-12  # tiny lambda for ridge regression
tol_allclose = 1e-8  # numerical tolerance

for _ in range(N):
    # Create rank-deficient X by repeating some columns
    X_base = np.random.randn(m, m)
    extra_cols = X_base[:, :n-m]  # repeat first n-m columns
    X = np.hstack([X_base, extra_cols])
    
    y = np.random.randn(m)
    
    # Pseudoinverse solution
    beta_pinverse = pseudoinverse_svd(X) @ y
    
    # Ridge regression solution using SVD (stable for nearly singular X)
    U, S, VT = np.linalg.svd(X, full_matrices=False)
    S_ridge = S / (S**2 + lam)
    beta_ridge = VT.T @ np.diag(S_ridge) @ U.T @ y
    
    if not np.allclose(beta_pinverse, beta_ridge, atol=tol_allclose):
        mismatch_count_rect += 1

print(f"(c) Number of mismatches in {N} rectangular matrix trials: {mismatch_count_rect}")
