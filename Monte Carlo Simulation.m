%% Monte Carlo European Call Option Pricer

clear;
clc;
close all;
% Set seed for reproducibility (optional)
rng('default'); 

%  Model Parameters 
S0 = 100;    % Initial stock price
r = 0.02;    % Risk-free rate
sigma = 0.30;% Volatility
T = 1;       % Time to maturity (in years)
N = 250;     % Number of time steps (for path-dependent method)

K_vec = [100, 102];         % Vector of strike prices
M_vec = [20000, 40000, 60000]; % Vector of trajectory counts

% 2. Run Simulations

fprintf('--- Monte Carlo vs. Black-Scholes --- \n');
fprintf('Params: S0=%.2f, r=%.2f, sigma=%.2f, T=%.2f, N=%d\n\n', S0, r, sigma, T, N);
fprintf('%-12s | %-6s | %-15s | %-15s | %-15s\n', ...
    'Trajectories', 'Strike', 'MC (Fast)', 'MC (Path)', 'Black-Scholes');
fprintf(repmat('-', 1, 80));
fprintf('\n');

% Loop over all simulation counts
for M = M_vec
    % Loop over all strike prices
    for K = K_vec
        
        % 1. Calculate Theoretical (Black-Scholes) Price
        bs_price = blackScholesCall(S0, K, T, r, sigma);
        
        % 2. Calculate Monte Carlo (Fast, Endpoint-Only)
        mc_price_fast = monteCarloCall_Fast(S0, K, T, r, sigma, M);
        
        % 3. Calculate Monte Carlo (Slow, Full Path)
        % This will be noticeably slower, especially at M=60,000
        mc_price_path = monteCarloCall_Path(S0, K, T, r, sigma, M, N);
        
        % 4. Print the results
        fprintf('%-12d | %-6.0f | %-15.4f | %-15.4f | %-15.4f\n', ...
            M, K, mc_price_fast, mc_price_path, bs_price);
    end
end

% --- 3. Local Helper Functions ---
% (These must be at the end of the script file)

function price = blackScholesCall(S, K, T, r, v)
    % Calculates the theoretical Black-Scholes price for a European call
    
    % Handle edge cases
    if T <= 0 || v <= 0
        price = max(0, S - K * exp(-r * T));
        return;
    end
    
    d1 = (log(S/K) + (r + 0.5 * v^2) * T) / (v * sqrt(T));
    d2 = d1 - v * sqrt(T);
    
    N_d1 = 0.5 * (1 + erf(d1 / sqrt(2)));
    N_d2 = 0.5 * (1 + erf(d2 / sqrt(2)));
    
    price = (S * N_d1) - (K * exp(-r * T) * N_d2);
end

function price = monteCarloCall_Fast(S0, K, T, r, sigma, M)
    
    % 1. Generate M random standard normal variables
    Z = randn(M, 1);
    
    % 2. Calculate M final stock prices (S_T) in a single vectorized step
    ST = S0 * exp((r - 0.5 * sigma^2) * T + sigma * sqrt(T) * Z);
    
    % 3. Calculate payoffs
    payoffs = max(ST - K, 0);
    
    % 4. Discount average payoff to get option price
    price = exp(-r * T) * mean(payoffs);
end

function price = monteCarloCall_Path(S0, K, T, r, sigma, M, N)
    % Inefficient (but requested) Monte Carlo: Simulates the full path
    
    dt = T / N;
    
    % 1. Pre-allocate price matrix: M rows (paths), N+1 columns (steps)
    S = zeros(M, N + 1);
    S(:, 1) = S0; % Set all initial prices to S0
    
    % 2. Generate all random numbers at once for efficiency
    Z = randn(M, N);
    
    % 3. Simulate all paths step-by-step (vectorized by path)
    drift = (r - 0.5 * sigma^2) * dt;
    diffusion = sigma * sqrt(dt);
    
    for j = 1:N
        S(:, j+1) = S(:, j) .* exp(drift + diffusion * Z(:, j));
    end
    
    % 4. Get final prices
    ST = S(:, end); 
    
    % 5. Calculate payoffs
    payoffs = max(ST - K, 0);
    
    % 6. Discount average payoff
    price = exp(-r * T) * mean(payoffs);
end