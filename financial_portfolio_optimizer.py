import streamlit as st  # Streamlit for building the web app interface
import pandas as pd  # Pandas for data manipulation and analysis
import numpy as np  # Numpy for numerical computations
import yfinance as yf  # Yahoo Finance API to fetch financial data
import matplotlib.pyplot as plt  # Matplotlib for plotting graphs
from matplotlib import cm  # Colormap utilities for visualization
from scipy.optimize import minimize, Bounds  # For portfolio optimization
from datetime import datetime  # To handle date and time operations

# -----------------------------
# Streamlit Configuration
# -----------------------------
# Set up the Streamlit app's page configuration with a title and a wide layout
st.set_page_config(page_title="Financial Portfolio Optimizer", layout="wide")

# -----------------------------
# Helper Functions
# -----------------------------

# Function to load CSV data containing stock symbols
def load_data(csv_file):
    try:
        df = pd.read_csv(csv_file)  # Read the CSV file into a DataFrame
        df.dropna(inplace=True)  # Remove any rows with missing values
        if 'Symbol' not in df.columns:  # Ensure 'Symbol' column is present
            st.error("CSV file must contain a 'Symbol' column.")
            st.stop()  # Stop execution if validation fails
        return df  # Return the cleaned DataFrame
    except Exception as e:
        st.error(f"Error loading CSV: {e}")  # Display error message if file reading fails
        st.stop()

# Function to fetch historical stock price data for the provided tickers
def fetch_price_data(tickers, start_date, end_date):
    valid_tickers = [ticker.strip().upper() + ".NS" for ticker in tickers if isinstance(ticker, str) and ticker.strip()]  # Clean and format ticker symbols
    if not valid_tickers:
        st.error("No valid tickers found.")  # Alert user if ticker list is empty
        st.stop()

    data = yf.download(valid_tickers, start=start_date, end=end_date, progress=False)['Close']  # Download closing price data
    data.dropna(axis=1, inplace=True)  # Remove stocks with incomplete data
    if data.empty:
        st.error("No valid stock data found. Please check ticker symbols or connection.")  # Handle scenario with no data
        st.stop()
    return data  # Return the price data DataFrame

# Calculate the portfolio's expected return and volatility given weights
def calculate_portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.dot(weights, mean_returns)  # Expected portfolio return
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))  # Portfolio volatility (standard deviation)
    return returns, volatility  # Return both metrics

# Objective function to be minimized based on user's risk preference
def risk_objective(weights, mean_returns, cov_matrix, risk_preference):
    expected_return, volatility = calculate_portfolio_performance(weights, mean_returns, cov_matrix)  # Get portfolio metrics

    # Adjust the objective based on selected risk preference
    if risk_preference == 'Low':
        return volatility  # For low risk: minimize volatility
    elif risk_preference == 'Medium':
        return -((expected_return) / volatility)  # For medium risk: maximize Sharpe ratio
    else:  # High risk
        return -expected_return  # For high risk: maximize returns

# Function to find the optimal portfolio weights
def optimize_portfolio(mean_returns, cov_matrix, risk_preference):
    num_assets = len(mean_returns)  # Count of assets in the portfolio
    bounds = Bounds([0.0] * num_assets, [0.25] * num_assets)  # Allocation between 0% and 25% per asset
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Total weights must sum to 1
    initial_guess = np.array([1 / num_assets] * num_assets)  # Equal initial allocation

    # Minimize the risk objective to find the optimal weights
    result = minimize(
        risk_objective,
        initial_guess,
        args=(mean_returns, cov_matrix, risk_preference),
        method='SLSQP',  # Sequential Least Squares Programming
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1500, 'ftol': 1e-9, 'disp': False}
    )

    if not result.success:
        st.error(f"Optimization failed: {result.message}")  # Show error if optimization fails
        st.stop()

    return pd.Series(result.x, index=mean_returns.index)  # Return weights as a Series

# Plot the efficient frontier along with the optimized portfolio point
def plot_efficient_frontier_with_optimum(mean_returns, cov_matrix, optimal_weights):
    num_points = 100  # Number of points to simulate on the frontier
    target_returns = np.linspace(mean_returns.min(), mean_returns.max(), num_points)  # Range of target returns
    frontier_volatility = []  # List to store volatility values

    for target in target_returns:
        # Constraints to achieve target return with sum of weights = 1
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: np.dot(w, mean_returns) - target}
        ]
        bounds = Bounds([0.0] * len(mean_returns), [0.25] * len(mean_returns))  # Respect allocation caps
        initial_guess = np.array([1 / len(mean_returns)] * len(mean_returns))  # Start with equal weights

        result = minimize(
            lambda w: calculate_portfolio_performance(w, mean_returns, cov_matrix)[1],  # Minimize volatility
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'disp': False}
        )
        frontier_volatility.append(result.fun if result.success else np.nan)  # Append result or NaN

    # Calculate optimized portfolio's performance
    portfolio_return, portfolio_volatility = calculate_portfolio_performance(
        optimal_weights.values, mean_returns, cov_matrix
    )

    # Plotting the efficient frontier and optimized portfolio
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(frontier_volatility, target_returns, 'o-', markersize=3, label='Efficient Frontier')
    ax.scatter(portfolio_volatility, portfolio_return, c='red', marker='*', s=200, label='Optimized Portfolio')
    ax.set_title("Efficient Frontier vs Optimized Portfolio")
    ax.set_xlabel("Volatility (Risk)")
    ax.set_ylabel("Expected Return")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)  # Add grid for better visualization

    return fig  # Return the plot

# Plot the growth of the portfolio over the investment horizon
def plot_portfolio_growth(investment_amount, portfolio_return, investment_years):
    future_value = investment_amount * (1 + portfolio_return) ** investment_years  # Future value of investment
    years = np.arange(0, investment_years + 1)  # Years for x-axis
    values = investment_amount * (1 + portfolio_return) ** years  # Portfolio value each year

    fig_growth, ax_growth = plt.subplots()
    ax_growth.plot(years, values, marker='o', color='teal')  # Plot investment growth
    ax_growth.set_title("Investment Growth Over Time")
    ax_growth.set_xlabel("Years")
    ax_growth.set_ylabel("Portfolio Value (INR)")
    ax_growth.grid(True, linestyle='--', alpha=0.7)  # Grid for clarity

    return fig_growth, future_value  # Return plot and future value

# Plot a pie chart to visualize portfolio allocation
def plot_portfolio_pie_chart(allocation):
    threshold = 2  # Minimum percentage to display separately
    large_alloc = allocation[allocation['Allocation (%)'] >= threshold]  # Major holdings
    small_alloc = allocation[allocation['Allocation (%)'] < threshold]  # Minor holdings grouped as 'Others'

    pie_labels = list(large_alloc['Stock']) + (['Others'] if not small_alloc.empty else [])  # Labels for pie chart
    pie_sizes = list(large_alloc['Allocation (INR)']) + ([small_alloc['Allocation (INR)'].sum()] if not small_alloc.empty else [])  # Size of slices

    if pie_sizes:
        fig_pie, ax_pie = plt.subplots(figsize=(10, 8))
        colors = cm.get_cmap('tab20')(np.linspace(0, 1, len(pie_labels)))  # Generate colors for segments
        ax_pie.pie(
            pie_sizes,
            labels=pie_labels,
            colors=colors,
            autopct='%1.1f%%',  # Show percentage allocation
            startangle=90,
            pctdistance=0.75,
            wedgeprops=dict(width=0.4, edgecolor='w')  # Donut chart style
        )
        ax_pie.set_title("Portfolio Allocation", fontsize=16, weight='bold')  # Add title
        return fig_pie  # Return the pie chart
    return None  # Return None if no valid data to plot

# -----------------------------
# Streamlit Application
# -----------------------------

# App title and introduction
st.title("📈 Financial Portfolio Optimizer")

# Explanation of Modern Portfolio Theory (MPT)
st.markdown("""
## 📚 Understanding Modern Portfolio Theory (MPT)
Modern Portfolio Theory (MPT), introduced by Harry Markowitz, emphasizes diversification to achieve optimal returns for a given level of risk.

### 🔑 Key Concepts:
- **Expected Return:** Weighted average return of assets.
- **Risk (Volatility):** Variation in returns measured by standard deviation.
- **Efficient Frontier:** Set of portfolios with the best return-risk combination.
- **Optimal Portfolio:** Portfolio that aligns with the investor's risk preference.

⚠️ *Note: Historical performance doesn't guarantee future results.*
""")

# Sidebar for user input
st.sidebar.header("User Input")
investment_amount = st.sidebar.number_input("Investment Amount (INR)", min_value=10000, step=1000, value=50000)  # Amount to invest
risk_preference = st.sidebar.selectbox("Risk Preference", ["Low", "Medium", "High"])  # Risk level
investment_years = st.sidebar.slider("Investment Horizon (Years)", 1, 20, 5)  # Investment duration

# Execute when the user clicks the submit button
if st.sidebar.button("Submit"):
    nifty50_df = load_data("ind_nifty50list.csv")  # Load NIFTY 50 tickers
    tickers = nifty50_df['Symbol'].dropna().unique().tolist()  # Extract ticker symbols

    end_date = datetime.today().strftime('%Y-%m-%d')  # Set end date to today
    start_date = (datetime.today() - pd.DateOffset(years=investment_years)).strftime('%Y-%m-%d')  # Calculate start date

    # Fetch stock price data
    with st.spinner("Fetching stock data..."):
        price_data = fetch_price_data(tickers, start_date, end_date)

    # Calculate daily returns and annualized statistics
    returns = np.log(price_data / price_data.shift(1)).dropna()  # Daily log returns
    mean_returns = returns.mean() * 252  # Annualized returns
    cov_matrix = returns.cov() * 252  # Annualized covariance

    # Optimize portfolio based on user inputs
    with st.spinner("Optimizing portfolio..."):
        optimal_weights = optimize_portfolio(mean_returns, cov_matrix, risk_preference)

    # Calculate portfolio performance metrics
    portfolio_return, portfolio_volatility = calculate_portfolio_performance(
        optimal_weights.values, mean_returns, cov_matrix
    )

    # Display portfolio allocation
    allocation = pd.DataFrame({
        'Stock': optimal_weights.index,
        'Weight': optimal_weights.values,
        'Allocation (INR)': optimal_weights.values * investment_amount,
        'Allocation (%)': optimal_weights.values * 100
    }).query("`Allocation (%)` > 0.01").sort_values(by='Allocation (%)', ascending=False).reset_index(drop=True)

    st.subheader("📃 Selected Stocks & Allocation")
    st.dataframe(allocation, use_container_width=True)  # Show allocation table

    # Plot and display portfolio allocation pie chart
    st.subheader("🥧 Portfolio Allocation Pie Chart")
    pie_chart = plot_portfolio_pie_chart(allocation)
    if pie_chart:
        st.pyplot(pie_chart)

    # Plot and display the efficient frontier
    st.subheader("💹 Efficient Frontier & Optimized Portfolio")
    with st.spinner("Generating efficient frontier..."):
        fig_frontier = plot_efficient_frontier_with_optimum(
            mean_returns.loc[optimal_weights.index],
            cov_matrix.loc[optimal_weights.index, optimal_weights.index],
            optimal_weights
        )
        st.pyplot(fig_frontier)

    # Plot and display investment growth projection
    st.subheader("📈 Investment Growth Over Time")
    fig_growth, future_value = plot_portfolio_growth(investment_amount, portfolio_return, investment_years)

    st.write(f"Expected Portfolio Return: **{portfolio_return * 100:.2f}%** per annum")
    st.write(f"Portfolio Volatility: **{portfolio_volatility * 100:.2f}%**")
    st.write(f"Projected Value After {investment_years} Years: **INR {future_value:,.2f}**")

    st.pyplot(fig_growth)
