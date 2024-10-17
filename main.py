import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import datetime
import seaborn as sns
import warnings

# **New Imports for Machine Learning**
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning,
                        message="The 'unit' keyword in TimedeltaIndex construction is deprecated")
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

# Fetch data with error handling
try:
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    data = data.ffill()
except Exception as e:
    print(f"Error fetching data: {e}")
    exit()

returns = data.pct_change().dropna()

# Check for NaNs and zeros in returns
if returns.isnull().values.any():
    print("Data contains NaNs. Dropping NaNs.")
    returns = returns.dropna()

# Step 2: Calculate covariance matrix
cov_matrix = returns.cov()
num_assets = len(tickers)
risk_free_rate = 0.05  # 5% annual risk-free rate

# **Machine Learning for Return Forecasting**

# Function to create lagged features
def create_features(asset_returns, lags=5):
    features = pd.DataFrame(index=asset_returns.index)
    for i in range(1, lags + 1):
        features[f'lag_{i}'] = asset_returns.shift(i)
    return features

# Prepare predicted returns
predicted_returns = pd.Series(index=tickers, dtype=float)

for ticker in tickers:
    # Prepare features and labels
    asset_returns = returns[ticker]
    features = create_features(asset_returns)
    labels = asset_returns.shift(-1)  # Predict next day's return

    # Combine features and labels and drop NaNs
    df_ml = pd.concat([features, labels], axis=1)
    df_ml.dropna(inplace=True)

    # Check if there is enough data after dropping NaNs
    if df_ml.empty or len(df_ml) < 10:
        print(f"Not enough data to train the model for {ticker}. Skipping.")
        predicted_returns[ticker] = np.nan
        continue

    # Separate features and labels
    labels = df_ml.iloc[:, -1]
    features = df_ml.iloc[:, :-1]

    # Keep track of the feature columns used during training
    feature_columns = features.columns

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, shuffle=False
    )

    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Prepare the last features for prediction using the same feature columns
    last_features = create_features(asset_returns).iloc[[-1]]
    last_features = last_features[feature_columns]

    # Check if last_features contain NaNs
    if last_features.isnull().any().any():
        print(f"Cannot predict return for {ticker} due to insufficient data in last features.")
        predicted_returns[ticker] = np.nan
        continue

    # Predict the next period return
    predicted_return = model.predict(last_features)[0]
    predicted_returns[ticker] = predicted_return

print("\nPredicted Returns:")
print(predicted_returns.dropna())

# Filter tickers with valid predicted returns
valid_tickers = predicted_returns.dropna().index.tolist()
if not valid_tickers:
    raise ValueError("No valid predicted returns available for optimization.")

# Update variables with valid tickers
predicted_mean_returns = predicted_returns.dropna() * 252  # Annualize predicted returns
cov_matrix = returns[valid_tickers].cov()
num_assets = len(valid_tickers)
tickers = valid_tickers

# Update constraints, bounds, and initial guess
constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
bounds = tuple((0.01, 1) for _ in range(num_assets))  # Minimum weight of 1%
init_guess = num_assets * [1. / num_assets]

# Function to calculate portfolio performance
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.dot(weights, mean_returns)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    return returns, volatility

# Optimization options with adjusted 'eps'
options = {'eps': 1e-10}

# **A. Minimum Variance Portfolio**

def min_variance(weights):
    return portfolio_performance(weights, predicted_mean_returns, cov_matrix)[1]  # Return volatility

opt_min_variance = minimize(
    min_variance,
    init_guess,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints,
    options=options
)

min_var_weights = opt_min_variance.x
min_var_return, min_var_volatility = portfolio_performance(
    min_var_weights, predicted_mean_returns, cov_matrix
)
min_var_sharpe = (
    (min_var_return - risk_free_rate) / min_var_volatility
    if min_var_volatility != 0 else np.nan
)

# **B. Maximum Return Portfolio**

def neg_return(weights):
    return -portfolio_performance(weights, predicted_mean_returns, cov_matrix)[0]

opt_max_return = minimize(
    neg_return,
    init_guess,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints,
    options=options
)

max_return_weights = opt_max_return.x
max_return_return, max_return_volatility = portfolio_performance(
    max_return_weights, predicted_mean_returns, cov_matrix
)
max_return_sharpe = (
    (max_return_return - risk_free_rate) / max_return_volatility
    if max_return_volatility != 0 else np.nan
)

# **C. Maximum Sharpe Ratio Portfolio**

def neg_sharpe_ratio(weights):
    p_ret, p_vol = portfolio_performance(weights, predicted_mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_vol if p_vol != 0 else 0

opt_max_sharpe = minimize(
    neg_sharpe_ratio,
    init_guess,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints,
    options=options
)

max_sharpe_weights = opt_max_sharpe.x
max_sharpe_return, max_sharpe_volatility = portfolio_performance(
    max_sharpe_weights, predicted_mean_returns, cov_matrix
)
max_sharpe_ratio = (
    (max_sharpe_return - risk_free_rate) / max_sharpe_volatility
    if max_sharpe_volatility != 0 else np.nan
)

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
mc_return, mc_volatility = portfolio_performance(
    market_cap_weights, predicted_mean_returns, cov_matrix
)
mc_sharpe = (
    (mc_return - risk_free_rate) / mc_volatility
    if mc_volatility != 0 else np.nan
)

# **Display Portfolio Weights and Performance**

def display_portfolio(weights, portfolio_name):
    print(f"\n{portfolio_name} Portfolio Weights:")
    for ticker, weight in zip(tickers, weights):
        print(f"{ticker}: {weight * 100:.2f}%")
    p_return, p_volatility = portfolio_performance(weights, predicted_mean_returns, cov_matrix)
    p_sharpe = (p_return - risk_free_rate) / p_volatility if p_volatility != 0 else np.nan
    print(f"Expected Annual Return (Predicted): {p_return * 100:.2f}%")
    print(f"Annual Volatility (Risk): {p_volatility * 100:.2f}%")
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
        weights = np.random.uniform(0.01, 1, num_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)
        p_return, p_volatility = portfolio_performance(weights, mean_returns, cov_matrix)
        p_sharpe = (p_return - risk_free_rate) / p_volatility if p_volatility != 0 else np.nan
        results[0, i] = p_volatility * 100  # Convert to percentage
        results[1, i] = p_return * 100      # Convert to percentage
        results[2, i] = p_sharpe
    return results, weights_record

results, _ = random_portfolios(50000, predicted_mean_returns, cov_matrix, risk_free_rate)

# Filter out any portfolios with NaN values
valid_results = results[:, ~np.isnan(results).any(axis=0)]

# Plotting the Efficient Frontier and Portfolios
plt.figure(figsize=(12, 8))
plt.scatter(valid_results[0], valid_results[1], c=valid_results[2], cmap='viridis', s=2, alpha=0.5)
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility (%)')
plt.ylabel('Expected Return (%)')
plt.title('Efficient Frontier with Predicted Returns')

# Plotting the portfolios as circles with adjusted sizes
plt.scatter(
    min_var_volatility * 100, min_var_return * 100,
    color='red', marker='o', s=100, label='Minimum Variance'
)
plt.scatter(
    max_return_volatility * 100, max_return_return * 100,
    color='blue', marker='o', s=100, label='Maximum Return'
)
plt.scatter(
    max_sharpe_volatility * 100, max_sharpe_return * 100,
    color='green', marker='o', s=100, label='Maximum Sharpe Ratio'
)
plt.scatter(
    mc_volatility * 100, mc_return * 100,
    color='purple', marker='o', s=100, label='Market Cap Weighted'
)

plt.legend(labelspacing=0.8)
plt.savefig('efficient_frontier_predicted.png')
plt.close()

# **Monte Carlo Simulation and VaR/CVaR Calculations**

def monte_carlo_simulation(mean_returns, cov_matrix, weights, num_simulations=10000, num_days=252):
    daily_mean_returns = mean_returns / 252
    daily_cov_matrix = cov_matrix / 252
    port_returns = []
    for _ in range(num_simulations):
        # Simulate daily returns over the year
        daily_returns = np.random.multivariate_normal(
            daily_mean_returns, daily_cov_matrix, num_days
        )
        # Calculate portfolio daily returns
        port_daily_returns = np.sum(daily_returns * weights, axis=1)
        # Compound daily returns to get annual return
        port_cumulative_return = np.prod(1 + port_daily_returns) - 1
        port_returns.append(port_cumulative_return)
    return np.array(port_returns)

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
    # Monte Carlo Simulation
    portfolio_returns = monte_carlo_simulation(predicted_mean_returns, cov_matrix, weights)
    # Output summary statistics
    print(f"\n{name} Portfolio Simulated Returns:")
    print(f"Mean Return: {np.mean(portfolio_returns) * 100:.2f}%")
    print(f"Median Return: {np.median(portfolio_returns) * 100:.2f}%")
    print(f"Minimum Return: {np.min(portfolio_returns) * 100:.2f}%")
    print(f"Maximum Return: {np.max(portfolio_returns) * 100:.2f}%")
    # Calculate VaR and CVaR
    var, cvar = calculate_var_cvar(portfolio_returns, confidence_level=0.95)
    print(f"\n{name} Portfolio:")
    print(f"Value at Risk (95% confidence): {var * 100:.2f}%")
    print(f"Conditional Value at Risk (95% confidence): {cvar * 100:.2f}%")
    # Plot the distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(portfolio_returns * 100, bins=50, kde=True, color='skyblue')
    plt.title(f'Distribution of Simulated Annual Returns ({name} Portfolio)')
    plt.xlabel('Simulated Annual Return (%)')
    plt.ylabel('Frequency')
    # Save the plot
    filename = f'{name.lower().replace(" ", "_")}_returns_distribution_predicted.png'
    plt.savefig(filename)
    plt.close()
