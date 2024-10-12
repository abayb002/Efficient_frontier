import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import datetime
import seaborn as sns
import warnings

# Suppress warnings
# warnings.filterwarnings("ignore", category=RuntimeWarning)

# Set the style for seaborn
sns.set(style='whitegrid')

# Step 1: Fetch historical data
tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMD', 'META']

# Adjust the end date to today's date
end_date = datetime.today().strftime('%Y-%m-%d')
data = yf.download(tickers, start='2015-01-01', end=end_date)['Adj Close']

# Handle missing data by forward filling
data = data.ffill()

returns = data.pct_change().dropna()

# Step 2: Calculate mean returns and covariance matrix
mean_returns = returns.mean()
cov_matrix = returns.cov()
num_assets = len(tickers)
risk_free_rate = 0.05  # Keep at 5% as per your request

# Function to calculate portfolio performance
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.dot(weights, mean_returns) * 252
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    return returns, volatility

# Constraints and bounds
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0.01, 1) for _ in range(num_assets))  # Minimum weight of 1%

# Initial guess
init_guess = num_assets * [1. / num_assets]

# Optimization options with adjusted 'eps'
options = {'eps': 1e-10}

# **A. Minimum Variance Portfolio**

def min_variance(weights):
    return portfolio_performance(weights, mean_returns, cov_matrix)[1]  # Return volatility

opt_min_variance = minimize(
    min_variance,
    init_guess,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints,
    options=options
)
min_var_weights = opt_min_variance.x
min_var_return, min_var_volatility = portfolio_performance(min_var_weights, mean_returns, cov_matrix)
min_var_sharpe = (min_var_return - risk_free_rate) / min_var_volatility

# **B. Maximum Return Portfolio**

def neg_return(weights):
    return -portfolio_performance(weights, mean_returns, cov_matrix)[0]  # Negative return for minimization

opt_max_return = minimize(
    neg_return,
    init_guess,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints,
    options=options
)
max_return_weights = opt_max_return.x
max_return_return, max_return_volatility = portfolio_performance(max_return_weights, mean_returns, cov_matrix)
max_return_sharpe = (max_return_return - risk_free_rate) / max_return_volatility

# **C. Maximum Sharpe Ratio Portfolio**

def neg_sharpe_ratio(weights):
    p_ret, p_vol = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_vol  # Negative Sharpe ratio for minimization

opt_max_sharpe = minimize(
    neg_sharpe_ratio,
    init_guess,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints,
    options=options
)
max_sharpe_weights = opt_max_sharpe.x
max_sharpe_return, max_sharpe_volatility = portfolio_performance(max_sharpe_weights, mean_returns, cov_matrix)
max_sharpe_ratio = (max_sharpe_return - risk_free_rate) / max_sharpe_volatility

# **Display Portfolio Weights and Performance**

def display_portfolio(weights, portfolio_name):
    print(f"\n{portfolio_name} Portfolio Weights:")
    for ticker, weight in zip(tickers, weights):
        print(f"{ticker}: {weight:.2%}")
    p_return, p_volatility = portfolio_performance(weights, mean_returns, cov_matrix)
    p_sharpe = (p_return - risk_free_rate) / p_volatility
    print(f"Expected Annual Return: {p_return:.2%}")
    print(f"Annual Volatility (Risk): {p_volatility:.2%}")
    print(f"Sharpe Ratio: {p_sharpe:.2f}")

display_portfolio(min_var_weights, "Minimum Variance")
display_portfolio(max_return_weights, "Maximum Return")
display_portfolio(max_sharpe_weights, "Maximum Sharpe Ratio")

# **Plot Efficient Frontier and Portfolios**

def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    results = np.zeros((3, num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.uniform(0.01, 1, num_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)
        p_return, p_volatility = portfolio_performance(weights, mean_returns, cov_matrix)
        p_sharpe = (p_return - risk_free_rate) / p_volatility
        results[0, i] = p_volatility
        results[1, i] = p_return
        results[2, i] = p_sharpe
    return results, weights_record

results, _ = random_portfolios(50000, mean_returns, cov_matrix, risk_free_rate)

# Plotting the Efficient Frontier and Portfolios
plt.figure(figsize=(12, 8))
plt.scatter(results[0], results[1], c=results[2], cmap='viridis', s=2, alpha=0.5)
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility (Standard Deviation)')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier')

# Plotting the portfolios as circles with adjusted sizes
plt.scatter(min_var_volatility, min_var_return, color='red', marker='o', s=100, label='Minimum Variance')
plt.scatter(max_return_volatility, max_return_return, color='blue', marker='o', s=100, label='Maximum Return')
plt.scatter(max_sharpe_volatility, max_sharpe_return, color='green', marker='o', s=100, label='Maximum Sharpe Ratio')

plt.legend(labelspacing=0.8)
plt.show()

# **Monte Carlo Simulation and VaR/CVaR Calculations**

def monte_carlo_simulation(mean_returns, cov_matrix, weights, num_simulations=10000):
    port_returns = []
    for _ in range(num_simulations):
        # Simulate daily returns
        simulated_returns = np.random.multivariate_normal(mean_returns, cov_matrix)
        # Calculate portfolio return
        port_return = np.sum(simulated_returns * weights)
        port_returns.append(port_return)
    port_returns = np.array(port_returns)
    return port_returns

def calculate_var_cvar(portfolio_returns, confidence_level=0.95):
    sorted_returns = np.sort(portfolio_returns)
    index = int((1 - confidence_level) * len(sorted_returns))
    var = -sorted_returns[index]
    cvar = -sorted_returns[:index].mean()
    return var, cvar

# **Calculate and Plot for Each Portfolio**

portfolios = {
    'Minimum Variance': min_var_weights,
    'Maximum Return': max_return_weights,
    'Maximum Sharpe Ratio': max_sharpe_weights
}

for name, weights in portfolios.items():
    # Monte Carlo Simulation
    portfolio_returns = monte_carlo_simulation(mean_returns, cov_matrix, weights)
    # Annualize the returns
    portfolio_returns_annualized = portfolio_returns * 252
    # Calculate VaR and CVaR
    var, cvar = calculate_var_cvar(portfolio_returns_annualized, confidence_level=0.95)
    print(f"\n{name} Portfolio:")
    print(f"Value at Risk (95% confidence): {var:.2f}%")
    print(f"Conditional Value at Risk (95% confidence): {cvar:.2f}%")
    # Plot the distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(portfolio_returns_annualized, bins=50, kde=True, color='skyblue')
    plt.title(f'Distribution of Simulated Annual Returns ({name} Portfolio)')
    plt.xlabel('Simulated Annual Return (%)')
    plt.ylabel('Frequency')
    plt.show()
