# streamlit run main.py

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import datetime
import seaborn as sns
import warnings
import streamlit as st

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning,
                        message="The 'unit' keyword in TimedeltaIndex construction is deprecated")
warnings.filterwarnings("ignore", category=DeprecationWarning,
                        message="Pyarrow will become a required dependency")

# Set the style for seaborn
sns.set(style='whitegrid')

# Set Streamlit page configuration
st.set_page_config(page_title="Efficient Frontier Portfolio Optimization", layout="wide")

# **Title and Description**
st.title("Efficient Frontier Portfolio Optimization")
st.markdown("""
This app allows you to perform portfolio optimization based on Modern Portfolio Theory.
Enter the stock tickers, select the date range, and adjust parameters to analyze different portfolios.
""")

# **Step 1: User Input for Tickers, Dates, and Benchmark**

# Get user input for tickers
tickers_input = st.text_input("Enter tickers separated by commas (e.g., AAPL, MSFT, GOOGL):",
                              "TSM, AVGO, JPM, NVO, UNH, HD, BAC, KO, PEP, BX, V, AXP, CAT, TXN, GOOGL, NVDA, META, AMZN")
tickers = [ticker.strip().upper() for ticker in tickers_input.split(',')]

# Get user input for start and end dates
start_date = st.date_input("Start Date", datetime(2015, 1, 1))
end_date = st.date_input("End Date", datetime.today())

# Ensure start_date is before end_date
if start_date > end_date:
    st.error("Error: Start date must be before end date.")
    st.stop()

# Allow user to input risk-free rate
risk_free_rate_input = st.number_input("Enter the annual risk-free rate as a percentage (e.g., 5 for 5%):",
                                       value=5.0, step=0.1)
risk_free_rate = risk_free_rate_input / 100

# **Add Maximum Weight Constraint**
max_weight_input = st.number_input("Enter the maximum weight per asset as a percentage (e.g., 20 for 20%):",
                                   value=20.0, step=1.0)
max_weight = max_weight_input / 100

# **Add User Input for Benchmark Ticker**
benchmark_ticker = st.text_input("Enter a benchmark ticker (e.g., SPY, QQQ):", "SPY")

# **Data Fetching with Caching**

@st.cache_data
def fetch_data(tickers, benchmark_ticker, start_date, end_date):
    # Validate tickers by checking if data can be fetched
    valid_tickers = []
    for ticker in tickers:
        try:
            test_data = yf.Ticker(ticker).history(period='1d')
            if not test_data.empty:
                valid_tickers.append(ticker)
            else:
                st.warning(f"No data found for ticker '{ticker}'. It will be skipped.")
        except Exception as e:
            st.warning(f"Error fetching data for ticker '{ticker}': {e}")

    tickers = valid_tickers
    if not tickers:
        st.error("No valid tickers provided. Please enter valid ticker symbols.")
        st.stop()

    # Fetch assets data
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Adj Close']
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()

    if data.empty:
        st.error("Asset data is empty after fetching. Please check the tickers and date range.")
        st.stop()

    # Fetch benchmark data
    try:
        # Ensure benchmark_ticker is a string, not a list
        benchmark_data = yf.download(benchmark_ticker, start=start_date, end=end_date, progress=False)['Adj Close']
        # Convert to Series if DataFrame with single column
        if isinstance(benchmark_data, pd.DataFrame):
            benchmark_data = benchmark_data.squeeze()
    except Exception as e:
        st.error(f"Error fetching benchmark data: {e}")
        st.stop()

    if benchmark_data.empty:
        st.error("Benchmark data is empty after fetching. Please check the benchmark ticker and date range.")
        st.stop()

    # Forward fill and drop NaNs
    data = data.ffill().dropna()
    benchmark_data = benchmark_data.ffill().dropna()

    # Make indices timezone-naive
    data.index = data.index.tz_localize(None)
    benchmark_data.index = benchmark_data.index.tz_localize(None)

    # Align dates using intersection
    common_dates = data.index.intersection(benchmark_data.index)
    if common_dates.empty:
        st.error("No overlapping dates between asset data and benchmark data.")
        st.stop()

    data = data.loc[common_dates]
    benchmark_data = benchmark_data.loc[common_dates]

    # Compute returns
    returns = data.pct_change().dropna()
    benchmark_returns = benchmark_data.pct_change().dropna()

    # Ensure indices are datetime and timezone-naive
    returns.index = pd.to_datetime(returns.index).tz_localize(None)
    benchmark_returns.index = pd.to_datetime(benchmark_returns.index).tz_localize(None)

    # Re-align after pct_change
    common_dates = returns.index.intersection(benchmark_returns.index)
    if common_dates.empty:
        st.error("No overlapping dates between asset returns and benchmark returns after returns calculation.")
        st.stop()

    returns = returns.loc[common_dates]
    benchmark_returns = benchmark_returns.loc[common_dates]

    return returns, benchmark_returns, tickers, benchmark_data

returns, benchmark_returns, tickers, benchmark_data = fetch_data(tickers, benchmark_ticker, start_date, end_date)

# Check for NaNs in returns
if returns.isnull().values.any():
    st.warning("Data contains NaNs. Dropping NaNs.")
    returns = returns.dropna()
    benchmark_returns = benchmark_returns.loc[returns.index]

# **Step 2: Calculate Mean Returns and Covariance Matrix**

# Ensure data alignment and convert to NumPy arrays
mean_returns = returns.mean().values  # Convert to NumPy array
cov_matrix = returns.cov().values     # Convert to NumPy array
returns_array = returns.values        # For use in simulations

num_assets = len(tickers)

# Function to calculate portfolio performance
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.dot(weights, mean_returns) * 252
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    return returns, volatility

# Function to compute portfolio variance
def portfolio_variance(weights, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))

# Function to compute portfolio daily returns
def compute_portfolio_returns(weights, returns_df):
    portfolio_returns = returns_df.dot(weights)
    return portfolio_returns

# **Display Covariance Matrix**

st.header("Covariance Matrix")
cov_matrix_df = pd.DataFrame(cov_matrix * 252, index=tickers, columns=tickers)  # Annualized covariance
st.dataframe(cov_matrix_df.style.background_gradient(cmap='coolwarm').format("{:.4f}"))

# Constraints and bounds
constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
bounds = tuple((0, max_weight) for _ in range(num_assets))  # Bounds between 0% and max_weight

# Initial guess
init_guess = num_assets * [1. / num_assets]

# Optimization options
options = {'maxiter': 1000, 'disp': False}

# **Optimization Functions**

def optimize_portfolios():
    # **A. Minimum Variance Portfolio**

    opt_min_variance = minimize(
        portfolio_variance,
        init_guess,
        args=(cov_matrix,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options=options
    )

    if not opt_min_variance.success:
        st.warning("Optimization for Minimum Variance Portfolio did not converge.")

    min_var_weights = opt_min_variance.x
    min_var_return, min_var_volatility = portfolio_performance(min_var_weights, mean_returns, cov_matrix)
    min_var_sharpe = (min_var_return - risk_free_rate) / min_var_volatility

    # **B. Maximum Sharpe Ratio Portfolio**

    def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
        p_ret, p_vol = portfolio_performance(weights, mean_returns, cov_matrix)
        return -(p_ret - risk_free_rate) / p_vol

    opt_max_sharpe = minimize(
        neg_sharpe_ratio,
        init_guess,
        args=(mean_returns, cov_matrix, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options=options
    )

    if not opt_max_sharpe.success:
        st.warning("Optimization for Maximum Sharpe Ratio Portfolio did not converge.")

    max_sharpe_weights = opt_max_sharpe.x
    max_sharpe_return, max_sharpe_volatility = portfolio_performance(max_sharpe_weights, mean_returns, cov_matrix)
    max_sharpe_ratio = (max_sharpe_return - risk_free_rate) / max_sharpe_volatility

    # **C. Maximum Return Portfolio**

    def neg_portfolio_return(weights, mean_returns):
        return -np.dot(weights, mean_returns) * 252

    opt_max_return = minimize(
        neg_portfolio_return,
        init_guess,
        args=(mean_returns,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options=options
    )

    if not opt_max_return.success:
        st.warning("Optimization for Maximum Return Portfolio did not converge.")

    max_return_weights = opt_max_return.x
    max_return_return, max_return_volatility = portfolio_performance(max_return_weights, mean_returns, cov_matrix)
    max_return_sharpe = (max_return_return - risk_free_rate) / max_return_volatility

    # **D. Market Cap Weighted Portfolio**

    # Fetch market capitalization data
    market_caps = {}
    for ticker in tickers:
        try:
            stock_info = yf.Ticker(ticker).info
            market_caps[ticker] = stock_info.get('marketCap', 0)
        except Exception as e:
            st.warning(f"Error fetching market cap for ticker '{ticker}': {e}")
            market_caps[ticker] = 0

    # Handle any missing market cap data
    total_market_cap = sum(market_caps.values())
    if total_market_cap == 0:
        st.error("Market capitalization data not available for the provided tickers.")
        st.stop()

    market_cap_weights = np.array([market_caps[ticker] for ticker in tickers]) / total_market_cap

    # Ensure weights sum to 1
    market_cap_weights /= np.sum(market_cap_weights)

    # Apply maximum weight constraint
    market_cap_weights = np.minimum(market_cap_weights, max_weight)
    market_cap_weights /= np.sum(market_cap_weights)

    # Calculate performance
    mc_return, mc_volatility = portfolio_performance(market_cap_weights, mean_returns, cov_matrix)
    mc_sharpe = (mc_return - risk_free_rate) / mc_volatility

    portfolios = {
        'Minimum Variance': {
            'Weights': min_var_weights,
            'Return': min_var_return,
            'Volatility': min_var_volatility,
            'Sharpe': min_var_sharpe
        },
        'Maximum Sharpe Ratio': {
            'Weights': max_sharpe_weights,
            'Return': max_sharpe_return,
            'Volatility': max_sharpe_volatility,
            'Sharpe': max_sharpe_ratio
        },
        'Maximum Return': {
            'Weights': max_return_weights,
            'Return': max_return_return,
            'Volatility': max_return_volatility,
            'Sharpe': max_return_sharpe
        },
        'Market Cap Weighted': {
            'Weights': market_cap_weights,
            'Return': mc_return,
            'Volatility': mc_volatility,
            'Sharpe': mc_sharpe
        }
    }

    return portfolios

portfolios = optimize_portfolios()

# **Generate Random Portfolios for Efficient Frontier**

def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    results = np.zeros((3, num_portfolios))
    for i in range(num_portfolios):
        weights = np.random.dirichlet(np.ones(num_assets), size=1)[0]
        weights = np.minimum(weights, max_weight)
        weights /= np.sum(weights)
        p_return, p_volatility = portfolio_performance(weights, mean_returns, cov_matrix)
        p_sharpe = (p_return - risk_free_rate) / p_volatility
        results[0, i] = p_volatility * 100  # Convert to percentage
        results[1, i] = p_return * 100      # Convert to percentage
        results[2, i] = p_sharpe
    return results

@st.cache_data
def generate_efficient_frontier():
    results = random_portfolios(10000, mean_returns, cov_matrix, risk_free_rate)
    return results

results = generate_efficient_frontier()

# **Display Portfolio Weights and Performance**

def display_portfolio(weights, portfolio_name):
    st.subheader(f"{portfolio_name} Portfolio Weights")
    weights_df = pd.DataFrame({'Ticker': tickers, 'Weight (%)': weights * 100})
    st.table(weights_df)

    p_return, p_volatility = portfolio_performance(weights, mean_returns, cov_matrix)
    p_sharpe = (p_return - risk_free_rate) / p_volatility

    # Compute portfolio daily returns
    portfolio_returns = compute_portfolio_returns(weights, returns)
    # Align portfolio returns with benchmark returns
    portfolio_returns.index = pd.to_datetime(portfolio_returns.index).tz_localize(None)
    benchmark_returns.index = pd.to_datetime(benchmark_returns.index).tz_localize(None)
    common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
    portfolio_returns = portfolio_returns.loc[common_dates]
    benchmark_returns_aligned = benchmark_returns.loc[common_dates]

    # Compute beta
    covariance = portfolio_returns.cov(benchmark_returns_aligned)
    benchmark_variance = benchmark_returns_aligned.var()
    beta = covariance / benchmark_variance

    st.write(f"**Expected Annual Return:** {p_return*100:.2f}%")
    st.write(f"**Annual Volatility (Risk):** {p_volatility*100:.2f}%")
    st.write(f"**Sharpe Ratio:** {p_sharpe:.2f}")
    st.write(f"**Beta with respect to {benchmark_ticker}:** {beta:.2f}")

# **Display Portfolios**

st.header("Optimized Portfolios")

for name, data in portfolios.items():
    display_portfolio(data['Weights'], name)

# **Calculate Benchmark Performance and Sharpe Ratio**

benchmark_annual_return = benchmark_returns.mean() * 252 * 100  # Convert to annual percentage
benchmark_annual_volatility = benchmark_returns.std() * np.sqrt(252) * 100  # Convert to annual percentage
benchmark_sharpe = (benchmark_annual_return / 100 - risk_free_rate) / (benchmark_annual_volatility / 100)

st.subheader(f"{benchmark_ticker} Benchmark Performance")
st.write(f"**Expected Annual Return:** {benchmark_annual_return:.2f}%")
st.write(f"**Annual Volatility (Risk):** {benchmark_annual_volatility:.2f}%")
st.write(f"**Sharpe Ratio:** {benchmark_sharpe:.2f}")

# **Plot Efficient Frontier and Portfolios**

st.header("Efficient Frontier")

plt.figure(figsize=(12, 8))

# Plot the random portfolios
plt.scatter(results[0], results[1], c=results[2], cmap='viridis', s=2, alpha=0.5)
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility (%)')
plt.ylabel('Expected Return (%)')
plt.title('Efficient Frontier')

# Plot the optimized portfolios
for name, data in portfolios.items():
    plt.scatter(data['Volatility'] * 100, data['Return'] * 100, marker='o', s=100, label=name)

# Plot benchmark
plt.scatter(benchmark_annual_volatility, benchmark_annual_return, marker='D', color='red', s=100,
            label=f'{benchmark_ticker} Benchmark')

plt.legend(labelspacing=0.8)

# Adjust axis limits to include all data points
all_volatilities = np.concatenate([
    results[0],
    np.array([data['Volatility'] * 100 for data in portfolios.values()]),
    np.array([benchmark_annual_volatility])
])

all_returns = np.concatenate([
    results[1],
    np.array([data['Return'] * 100 for data in portfolios.values()]),
    np.array([benchmark_annual_return])
])

x_min = np.min(all_volatilities) * 0.9
x_max = np.max(all_volatilities) * 1.1
y_min = np.min(all_returns) * 0.9
y_max = np.max(all_returns) * 1.1

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

st.pyplot(plt.gcf())
plt.close()

# **Plot Cumulative Returns**

st.header("Cumulative Returns Over Time")

# Compute cumulative returns for each portfolio and benchmark
cumulative_returns = pd.DataFrame(index=returns.index)
cumulative_returns['Benchmark'] = (1 + benchmark_returns).cumprod()

for name, data in portfolios.items():
    weights = data['Weights']
    portfolio_returns = compute_portfolio_returns(weights, returns)
    cumulative_returns[name] = (1 + portfolio_returns).cumprod()

# Plot cumulative returns
fig, ax = plt.subplots(figsize=(12, 8))
for column in cumulative_returns.columns:
    ax.plot(cumulative_returns.index, cumulative_returns[column], label=column)

ax.set_title('Cumulative Returns of Portfolios and Benchmark')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Return')
ax.legend()
st.pyplot(fig)
plt.close(fig)

# **Monte Carlo Simulation and VaR/CVaR Calculations**

def monte_carlo_simulation(mean_returns, cov_matrix, weights, num_simulations=10000, num_days=252):
    port_returns = []
    mean_daily_returns = mean_returns
    cov_daily = cov_matrix
    for _ in range(num_simulations):
        daily_returns = np.random.multivariate_normal(mean_daily_returns, cov_daily, num_days)
        port_daily_returns = np.dot(daily_returns, weights)
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

# **Calculate and Display for Each Portfolio**

st.header("Monte Carlo Simulation and Risk Metrics")

for name, data in portfolios.items():
    with st.expander(f"{name} Portfolio Risk Metrics"):
        weights = data['Weights']
        # Monte Carlo Simulation using multivariate normal distribution
        portfolio_returns = monte_carlo_simulation(mean_returns, cov_matrix, weights)
        # The portfolio_returns are already annualized through compounding
        # Calculate VaR and CVaR
        var, cvar = calculate_var_cvar(portfolio_returns, confidence_level=0.95)
        st.write(f"**Value at Risk (95% confidence):** {var*100:.2f}%")
        st.write(f"**Conditional Value at Risk (95% confidence):** {cvar*100:.2f}%")
        # Plot the distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(portfolio_returns * 100, bins=50, kde=True, color='skyblue', ax=ax)
        ax.set_title(f'Distribution of Simulated Annual Returns ({name} Portfolio)')
        ax.set_xlabel('Simulated Annual Return (%)')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)
        plt.close(fig)
