# 📈 Financial Portfolio Optimizer

A Python-based financial portfolio optimization tool leveraging **Modern Portfolio Theory (MPT)** to help investors construct an optimized portfolio based on risk-return trade-offs. Built with **Streamlit** for an interactive user experience, it fetches stock data from **Yahoo Finance** and provides insights through visualizations like **efficient frontier plots, allocation charts, and projected investment growth**.

---

## 🚀 Project Overview

Investors often struggle to construct a portfolio that balances **returns and risk** effectively. This project simplifies the process by allowing users to:

- Input **investment amount, risk preference, and investment horizon**.
- Optimize a portfolio of **NIFTY 50 stocks** using **MPT** principles.
- Visualize key portfolio metrics such as:
  - **Stock allocations** (table and pie chart)
  - **Efficient Frontier plot**
  - **Projected investment growth**

The optimizer considers constraints like **maximum 25% allocation per stock** and ensures well-diversified portfolios.

---

## 🛠️ Installation

To run this project locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/portfolio-optimizer.git
   cd portfolio-optimizer
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

---

## 📌 Usage Instructions

1. Open the **Streamlit interface**.
2. Enter:
   - Investment Amount (INR)
   - Risk Preference (**Low**, **Medium**, or **High**)
   - Investment Horizon (Years)
3. Click **Submit**, and the optimizer will:
   - Fetch historical stock data.
   - Optimize the portfolio based on your risk level.
   - Display stock allocation, efficient frontier, and projected returns.

---

## ⚙️ Dependencies

This project requires:

- `streamlit` (for UI)
- `pandas`, `numpy` (for data processing)
- `yfinance` (to fetch stock data)
- `matplotlib` (for plotting graphs)
- `scipy` (for optimization)

Install them using:
```bash
pip install streamlit pandas numpy yfinance matplotlib scipy
```

---

## 📊 How It Works

### Step 1: Data Retrieval
- Loads NIFTY 50 stock symbols from a CSV file.
- Fetches historical stock prices using `yfinance`.

### Step 2: Portfolio Optimization
- Computes **expected return** and **risk (volatility)**.
- Uses **Scipy's minimize()** function for optimization:
  - **Low Risk** → Minimizes volatility.
  - **Medium Risk** → Maximizes Sharpe Ratio.
  - **High Risk** → Maximizes returns.

### Step 3: Visualization & Insights
- **Stock Allocation** (Table & Pie Chart)
- **Efficient Frontier Plot**
- **Investment Growth Projection** (Time Series)

---

## 🎯 Future Enhancements

- **Real-time data integration** (via paid APIs).
- **More stock indices** for diversification.
- **Advanced risk measures** beyond standard deviation.
- **Personalized recommendations** based on user profiles.

---

## 🐝 License

This project is open-source and available under the **MIT License**.

---

### ⭐ Contributions & Feedback

- Feel free to **fork**, **star**, or **raise issues**!
- Suggestions? Connect with me on **[LinkedIn](https://www.linkedin.com/in/yourprofile/)** or open a GitHub issue.

---

🔗 **GitHub Repository**: [https://github.com/sanchaligorde/portfolio-optimizer](https://github.com/sanchaligorde/portfolio-optimizer)

