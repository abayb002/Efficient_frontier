import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Step 1: Fetch historical data
tickers = ['AAPL', 'MSFT', 'GOOGL','TSLA','NVDA','AMD','META']  # Replace with your portfolio tickers
data = yf.download(tickers, start='2015-01-01', end='2024-01-01')['Adj Close']
returns = data.pct_change().dropna()

# Step 2: Calculate mean returns and covariance matrix
mean_returns = returns.mean()
cov_matrix = returns.cov()


# Step 3: Define portfolio statistics (expected return, variance, Sharpe ratio)
def portfolio_statistics(weights, mean_returns, cov_matrix, risk_free_rate=0.01):
    portfolio_return = np.sum(mean_returns * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return portfolio_return, portfolio_volatility, sharpe_ratio


# Step 4: Objective function to minimize (negative Sharpe ratio for maximization)
def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.01):
    return -portfolio_statistics(weights, mean_returns, cov_matrix, risk_free_rate)[2]


# Constraints: sum of weights = 1, individual weight bounds between 0 and 1
constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
bounds = tuple((0, 1) for asset in range(len(mean_returns)))

# Initial guess (equal allocation)
initial_weights = len(mean_returns) * [1. / len(mean_returns)]

# Step 5: Optimization
optimized_result = minimize(negative_sharpe_ratio, initial_weights, args=(mean_returns, cov_matrix),
                            method='SLSQP', bounds=bounds, constraints=constraints)

# Step 6: Optimal weights and portfolio stats
optimal_weights = optimized_result.x
portfolio_return, portfolio_volatility, sharpe_ratio = portfolio_statistics(optimal_weights, mean_returns, cov_matrix)

print(f"Optimal Weights: {optimal_weights}")
print(f"Expected Portfolio Return: {portfolio_return:.3f}")
print(f"Portfolio Volatility (Risk): {portfolio_volatility:.3f}")
print(f"Sharpe Ratio: {sharpe_ratio:.3f}")


# Step 7: Plotting Efficient Frontier (Optional)
def plot_efficient_frontier(mean_returns, cov_matrix, num_portfolios=10000, risk_free_rate=0.01):
    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_return, portfolio_volatility, sharpe_ratio = portfolio_statistics(weights, mean_returns, cov_matrix)
        results[0, i] = portfolio_volatility
        results[1, i] = portfolio_return
        results[2, i] = sharpe_ratio

    plt.scatter(results[0, :], results[1, :], c=results[2, :], cmap='YlGnBu', marker='o')
    plt.title('Efficient Frontier')
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.show()


plot_efficient_frontier(mean_returns, cov_matrix)
