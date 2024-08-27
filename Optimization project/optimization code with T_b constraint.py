from scipy.optimize import minimize
import math

# Constants and input parameters
T_min_required = 1.35
T_max_required = 2.5
K = 2126.9  # Bulk modulus MPa
G_rubber = 0.42  # MPa
A_isolator = 200 * 100  # mm^2
P_isolator = 3  # Pressure on isolator N/mm^2
M = 100000  # Kg
N_iso = 17
E_f = 50000  # Young's modulus of fibre
z = 0.36  # Zone factor
B_eff = 0.102  # Expected damping ratio of isolator
u = 0.49995  # Poisson's ratio of rubber
limit = 1000  # Number of iterations for summation

# Function to calculate E_e based on the provided expression
def calculate_E_e(G_rubber, A_isolator, T, a, b, alpha, beta, limit):
    summation_result = 0
    for i in range(1, limit + 1):
        for j in range(1, limit + 1):
            numerator = math.sin(i * math.pi / 2) ** 2 * math.sin(j * math.pi / 2) ** 2
            denominator = i ** 2 * j ** 2 * (
                    (i * math.pi / 2) ** 2 * (j * math.pi / 2 * a / b) ** 2 +
                    2 * (alpha * a / b) ** 2 +
                    (beta * a / 2) ** 2
            )
            term = numerator / denominator
            summation_result += term

    E_e = (192 * G_rubber * A_isolator ** 2) / ((math.pi ** 4) * (T ** 2)) * summation_result
    return E_e

# Objective function to minimize shear strain energy
def objective_function(x):
    L, B, T, N_r, N_f, T_r, T_f = x  # Variables: (Length, Width, Thickness, no. of rubber layer, no. of fibre layer, thickness of rubber layer, thickness of fibre layer)

    V_isolator = L * B * T
    L2 = T_r  # Assuming this was meant to be T_r based on the equation used later

    # Calculate alpha and beta
    alpha = math.sqrt(24 * G_rubber / (50000 * T_f * T))
    beta = math.sqrt(12 * G_rubber / (K * (T ** 2)))

    # Calculate E_e using the summation function
    E_e = calculate_E_e(G_rubber, A_isolator, T, L, B, alpha, beta, limit)
    print("Result of E_e:", E_e)

    K_eff = (A_isolator * G_rubber) / T_r
    print("k_eff:", K_eff)

    T_b = 2 * math.pi * math.sqrt(M / (N_iso * (K_eff * (10 ** 3))))
    print("T_b:", T_b)
    return T_b

# Functions to calculate S_ag_value based on soil type and T_b
def calculate_S_ag_h(T_b):
    if T_b < 0.10:
        return 1 + 15 * T_b
    elif 0.10 <= T_b <= 0.40:
        return 2.5
    elif 0.40 <= T_b <= 4.00:
        return 1 / T_b
    else:
        return 0.25

def calculate_S_ag_m(T_b):
    if T_b < 0.10:
        return 1 + 15 * T_b
    elif 0.10 <= T_b <= 0.55:
        return 2.5
    elif 0.55 <= T_b <= 4.00:
        return 1.36 / T_b
    else:
        return 0.34

def calculate_S_ag_s(T_b):
    if T_b < 0.10:
        return 1 + 15 * T_b
    elif 0.10 <= T_b <= 0.67:
        return 2.5
    elif 0.67 <= T_b <= 4.00:
        return 1.67 / T_b
    else:
        return 0.42

def get_S_ag_value(soil_type, T_b):
    if soil_type.lower() == "hard":
        return calculate_S_ag_h(T_b)
    elif soil_type.lower() == "medium":
        return calculate_S_ag_m(T_b)
    elif soil_type.lower() == "soft":
        return calculate_S_ag_s(T_b)
    else:
        raise ValueError("Invalid soil type. Please choose 'hard,' 'medium,' or 'soft'.")

# Constraint function for T_b
def constraint_T_b(x):
    T_b_values = [objective_function(x) for _ in range(10)]  # Calculate T_b values for 10 iterations
    min_T_b = min(T_b_values)
    max_T_b = max(T_b_values)
    return min_T_b - T_min_required, T_max_required - max_T_b


if __name__ == "__main__":
    # Input soil type
    soil_type = input("Enter the soil type ('hard,' 'medium,' or 'soft'): ")

    initial_guess = [150, 150, 150, 8, 0.1, 8, 0.1]  # Initial guess for optimization variables

    variable_bounds = [(50, 250), (50, 250), (50, 250), (2, 15), (0.025, 0.2), (2, 15), (0.025, 0.2)]  # Bounds for each variable

    # Run optimization
    optimization_result = minimize(objective_function, initial_guess, bounds=variable_bounds, constraints={'type': 'ineq', 'fun': constraint_T_b}, method='SLSQP', tol=1e-3, options={'maxiter': 10000})

    print("\nOptimized variables:", optimization_result.x)
    print("Optimized objective function value (Shear Strain Energy):", optimization_result.fun)
    print("Optimized T_b values:", [objective_function(optimization_result.x) for _ in range(10)])
