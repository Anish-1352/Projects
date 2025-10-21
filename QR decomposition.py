import numpy as np

A = np.random.rand(6, 4)
print ("original matrix")
print (A)
Q,R = np.linalg.qr(A)
print(" Decomposed Matrix Q")
print(Q)

print("Decomposed Matrix R")
print(R)

print("\n--- Verification ---")

verification = np.allclose(A, Q @ R)
print(f"Does Q @ R equal A?  --> {verification}")

identity_matrix = np.identity(4)
orthonormal = np.allclose(Q.T @ Q, identity_matrix)
print(f"Is Q.T @ Q the Identity Matrix? --> {orthonormal}")