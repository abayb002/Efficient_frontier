import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import datetime
import seaborn as sns
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning, message="The 'unit' keyword in TimedeltaIndex construction is deprecated")
warnings.filterwarnings("ignore", category=DeprecationWarning, message="Pyarrow will become a required dependency")

# Set the style for seaborn
sns.set(style='whitegrid')

# **Step 1: User Input for Tickers and Dates**

# Get user input for tickers
tickers_input = input("Enter tickers separated by commas (e.g., AAPL, MSFT, GOOGL): ")
tickers = [ticker.strip().upper() for ticker in tickers_input.split(',')]

# Validate tickers by checking if data can be fetched
valid_tickers = []
for ticker in tickers:
    try:
        test_data = yf.Ticker(ticker).history(period='1d')
        if not test_data.empty:
            valid_tickers.append(ticker)
        else:
            print(f"Warning: No data found for ticker '{ticker}'. It will be skipped.")
    except Exception as e:
        print(f"Error fetching data for ticker '{ticker}': {e}")

tickers = valid_tickers
if not tickers:
    raise ValueError("No valid tickers provided. Please restart and enter valid ticker symbols.")

# Get user input for start and end dates
def validate_date(date_text):
    try:
        datetime.strptime(date_text, '%Y-%m-%d')
        return date_text
    except ValueError:
        raise ValueError("Incorrect date format. Please enter the date in YYYY-MM-DD format.")

start_date = validate_date(input("Enter the start date in YYYY-MM-DD format (e.g., 2015-01-01): "))
end_date_input = input("Enter the end date in YYYY-MM-DD format or press Enter for today's date: ")
if end_date_input.strip() == '':
    end_date = datetime.today().strftime('%Y-%m-%d')
else:
    end_date = validate_date(end_date_input)

# Fetch data with error handling and suppress the progress bar
try:
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Adj Close']
    data = data.ffill()
except Exception as e:
    print(f"Error fetching data: {e}")
    exit()

returns = data.pct_change().dropna()

# Check for NaNs and zeros in returns
if returns.isnull().values.any():
    print("Data contains NaNs. Dropping NaNs.")
    returns = returns.dropna()

# **Step 2: Calculate Mean Returns and Covariance Matrix**

# Ensure data alignment and convert to NumPy arrays
mean_returns = returns.mean().values  # Convert to NumPy array
cov_matrix = returns.cov().values     # Convert to NumPy array
returns_array = returns.values        # For use in simulations

num_assets = len(tickers)

# Allow user to input risk-free rate
risk_free_rate_input = input("Enter the annual risk-free rate as a percentage (e.g., 5 for 5%): ")
try:
    risk_free_rate = float(risk_free_rate_input) / 100
except ValueError:
    print("Invalid input for risk-free rate. Please enter a numerical value.")
    exit()

# Function to calculate portfolio performance
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.dot(weights, mean_returns) * 252
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    return returns, volatility

# Constraints and bounds
constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
bounds = tuple((0, 1) for _ in range(num_assets))  # Bounds between 0% and 100%

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

if not opt_min_variance.success:
    print("Optimization for Minimum Variance Portfolio did not converge.")
    # Handle accordingly, e.g., exit or proceed with warnings

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

if not opt_max_return.success:
    print("Optimization for Maximum Return Portfolio did not converge.")

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

if not opt_max_sharpe.success:
    print("Optimization for Maximum Sharpe Ratio Portfolio did not converge.")

max_sharpe_weights = opt_max_sharpe.x
max_sharpe_return, max_sharpe_volatility = portfolio_performance(max_sharpe_weights, mean_returns, cov_matrix)
max_sharpe_ratio = (max_sharpe_return - risk_free_rate) / max_sharpe_volatility

# **D. Market Cap Weighted Portfolio**

# Fetch market capitalization data
market_caps = {}
for ticker in tickers:
    stock_info = yf.Ticker(ticker).info
    market_caps[ticker] = stock_info.get('marketCap', 0)

# Handle any missing market cap data
total_market_cap = sum(market_caps.values())
if total_market_cap == 0:
    raise ValueError("Market capitalization data not available for the provided tickers.")

market_cap_weights = np.array([market_caps[ticker] for ticker in tickers]) / total_market_cap

# Ensure weights sum to 1
market_cap_weights /= np.sum(market_cap_weights)

# Calculate performance
mc_return, mc_volatility = portfolio_performance(market_cap_weights, mean_returns, cov_matrix)
mc_sharpe = (mc_return - risk_free_rate) / mc_volatility

# **Display Portfolio Weights and Performance**

def display_portfolio(weights, portfolio_name):
    print(f"\n{portfolio_name} Portfolio Weights:")
    for ticker, weight in zip(tickers, weights):
        print(f"{ticker}: {weight*100:.2f}%")
    p_return, p_volatility = portfolio_performance(weights, mean_returns, cov_matrix)
    p_sharpe = (p_return - risk_free_rate) / p_volatility
    print(f"Expected Annual Return: {p_return*100:.2f}%")
    print(f"Annual Volatility (Risk): {p_volatility*100:.2f}%")
    print(f"Sharpe Ratio: {p_sharpe:.2f}")

display_portfolio(min_var_weights, "Minimum Variance")
display_portfolio(max_return_weights, "Maximum Return")
display_portfolio(max_sharpe_weights, "Maximum Sharpe Ratio")
display_portfolio(market_cap_weights, "Market Cap Weighted")

# **Plot Efficient Frontier and Portfolios**

def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    results = np.zeros((3, num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.uniform(0, 1, num_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)
        p_return, p_volatility = portfolio_performance(weights, mean_returns, cov_matrix)
        p_sharpe = (p_return - risk_free_rate) / p_volatility
        results[0, i] = p_volatility * 100  # Convert to percentage
        results[1, i] = p_return * 100      # Convert to percentage
        results[2, i] = p_sharpe
    return results, weights_record

results, _ = random_portfolios(50000, mean_returns, cov_matrix, risk_free_rate)

# Plotting the Efficient Frontier and Portfolios
plt.figure(figsize=(12, 8))
plt.scatter(results[0], results[1], c=results[2], cmap='viridis', s=2, alpha=0.5)
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility (%)')
plt.ylabel('Expected Return (%)')
plt.title('Efficient Frontier')

# Plotting the portfolios as circles with adjusted sizes
plt.scatter(min_var_volatility * 100, min_var_return * 100, color='red', marker='o', s=100, label='Minimum Variance')
plt.scatter(max_return_volatility * 100, max_return_return * 100, color='blue', marker='o', s=100, label='Maximum Return')
plt.scatter(max_sharpe_volatility * 100, max_sharpe_return * 100, color='green', marker='o', s=100, label='Maximum Sharpe Ratio')
plt.scatter(mc_volatility * 100, mc_return * 100, color='purple', marker='o', s=100, label='Market Cap Weighted')

plt.legend(labelspacing=0.8)
plt.savefig('efficient_frontier.png')
plt.close()

# **Monte Carlo Simulation and VaR/CVaR Calculations**

def monte_carlo_simulation(mean_returns, cov_matrix, weights, num_simulations=10000, num_days=252):
    port_returns = []
    # Use bootstrapping from historical returns
    historical_returns = returns_array  # Use the aligned returns array
    num_obs = historical_returns.shape[0]
    for _ in range(num_simulations):
        # Randomly sample days with replacement
        sampled_indices = np.random.randint(0, num_obs, num_days)
        sampled_returns = historical_returns[sampled_indices]
        # Calculate portfolio daily returns
        port_daily_returns = np.sum(sampled_returns * weights, axis=1)
        # Compound daily returns to get annual return
        port_cumulative_return = np.prod(1 + port_daily_returns) - 1
        port_returns.append(port_cumulative_return)
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
    'Maximum Sharpe Ratio': max_sharpe_weights,
    'Market Cap Weighted': market_cap_weights
}

for name, weights in portfolios.items():
    # Monte Carlo Simulation using bootstrapping
    portfolio_returns = monte_carlo_simulation(mean_returns, cov_matrix, weights)
    # The portfolio_returns are already annualized through compounding
    # Calculate VaR and CVaR
    var, cvar = calculate_var_cvar(portfolio_returns, confidence_level=0.95)
    print(f"\n{name} Portfolio:")
    print(f"Value at Risk (95% confidence): {var*100:.2f}%")
    print(f"Conditional Value at Risk (95% confidence): {cvar*100:.2f}%")
    # Plot the distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(portfolio_returns * 100, bins=50, kde=True, color='skyblue')
    plt.title(f'Distribution of Simulated Annual Returns ({name} Portfolio)')
    plt.xlabel('Simulated Annual Return (%)')
    plt.ylabel('Frequency')
    # Save the plot
    filename = f'{name.lower().replace(" ", "_")}_returns_distribution.png'
    plt.savefig(filename)
    plt.close()
