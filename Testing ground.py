import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# Step 1: Get the user's input for stock tickers
def get_stock_tickers():
    print("Please enter the stock tickers you want to analyze, separated by commas (e.g., AAPL, MSFT, GOOGL):")
    tickers_input = input()
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(",")]
    return tickers


# Step 2: Download stock data from yfinance
def fetch_stock_data(tickers):
    data = yf.download(tickers, start='2015-01-01', end='2024-01-01')['Adj Close']
    returns = data.pct_change().dropna()
    return returns


# Step 3: Portfolio statistics calculation
def portfolio_statistics(weights, mean_returns, cov_matrix, risk_free_rate=0.01):
    portfolio_return = np.sum(mean_returns * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return portfolio_return, portfolio_volatility, sharpe_ratio


# Step 4: Objective function for Sharpe ratio maximization
def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.01):
    return -portfolio_statistics(weights, mean_returns, cov_matrix, risk_free_rate)[2]


# Step 5: Optimize the portfolio
def optimize_portfolio(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    result = minimize(negative_sharpe_ratio, num_assets * [1. / num_assets], args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result


# Step 6: Efficient frontier plot
def plot_efficient_frontier(mean_returns, cov_matrix, num_portfolios=10000):
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


# Step 7: Run the analysis for multiple portfolios
def compare_portfolios():
    while True:
        tickers = get_stock_tickers()
        returns = fetch_stock_data(tickers)
        mean_returns = returns.mean()
        cov_matrix = returns.cov()

        # Optimize portfolio and calculate statistics
        result = optimize_portfolio(mean_returns, cov_matrix)
        optimal_weights = result.x
        portfolio_return, portfolio_volatility, sharpe_ratio = portfolio_statistics(optimal_weights, mean_returns,
                                                                                    cov_matrix)

        print(f"\nOptimal Weights for Portfolio: {dict(zip(tickers, optimal_weights))}")
        print(f"Expected Portfolio Return: {portfolio_return:.3f}")
        print(f"Portfolio Volatility (Risk): {portfolio_volatility:.3f}")
        print(f"Sharpe Ratio: {sharpe_ratio:.3f}")

        # Plot efficient frontier
        plot_efficient_frontier(mean_returns, cov_matrix)

        # Ask the user if they want to compare another portfolio
        compare_again = input("\nWould you like to compare another portfolio? (yes/no): ").strip().lower()
        if compare_again != 'yes':
            break


# Run the program
if __name__ == "__main__":
    compare_portfolios()
