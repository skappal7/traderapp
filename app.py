import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import logging

logging.basicConfig(level=logging.ERROR)

# List of popular stocks and S&P 500 sector indices
popular_stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V']
sp500_sectors = {
    'S&P 500': '^GSPC',
    'Information Technology': 'XLK',
    'Health Care': 'XLV',
    'Financials': 'XLF',
    'Consumer Discretionary': 'XLY',
    'Communication Services': 'XLC',
    'Industrials': 'XLI',
    'Consumer Staples': 'XLP',
    'Energy': 'XLE',
    'Utilities': 'XLU',
    'Real Estate': 'XLRE',
    'Materials': 'XLB'
}

# Function to fetch stock data
def get_stock_data(symbol, start_date, end_date):
    return yf.download(symbol, start=start_date, end=end_date)

# Function to create candlestick chart with indicators
def plot_stock_data(data, indicators):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, subplot_titles=('Stock Price', 'Volume'),
                        row_heights=[0.7, 0.3])
    
    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'],
                                 low=data['Low'], close=data['Close'], name='Candlestick'),
                  row=1, col=1)

    colors = ['green' if row['Open'] - row['Close'] >= 0 
              else 'red' for index, row in data.iterrows()]
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color=colors),
                  row=2, col=1)

    for indicator in indicators:
        fig.add_trace(go.Scatter(x=data.index, y=data[indicator], name=indicator),
                      row=1, col=1)

    fig.update_layout(height=800, width=1200, showlegend=True,
                      xaxis_rangeslider_visible=False)
    return fig

# Function to calculate technical indicators
def add_indicators(data):
    data['SMA20'] = ta.trend.sma_indicator(data['Close'], window=20)
    data['SMA120'] = ta.trend.sma_indicator(data['Close'], window=120)
    data['EMA20'] = ta.trend.ema_indicator(data['Close'], window=20)
    data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
    data['VWAP'] = ta.volume.volume_weighted_average_price(data['High'], data['Low'], data['Close'], data['Volume'])
    return data

# Function for trends and momentum strategy
def trends_momentum_strategy(data):
    data['Signal'] = np.where(data['SMA20'] > data['SMA120'], 1, 0)
    data['Returns'] = data['Close'].pct_change()
    data['Strategy_Returns'] = data['Signal'].shift(1) * data['Returns']
    return data['Strategy_Returns'].cumsum().iloc[-1]

# Function for mean reversion strategy
def mean_reversion_strategy(data):
    data['Signal'] = np.where(data['Close'] < data['SMA20'], 1, 0)
    data['Returns'] = data['Close'].pct_change()
    data['Strategy_Returns'] = data['Signal'].shift(1) * data['Returns']
    return data['Strategy_Returns'].cumsum().iloc[-1]

# Function for VWAP strategy
def vwap_strategy(data):
    data['Signal'] = np.where(data['Close'] < data['VWAP'], 1, 0)
    data['Returns'] = data['Close'].pct_change()
    data['Strategy_Returns'] = data['Signal'].shift(1) * data['Returns']
    return data['Strategy_Returns'].cumsum().iloc[-1]

# Function for statistical arbitrage strategy
def statistical_arbitrage_strategy(data1, data2):
    spread = data1['Close'] - data2['Close']
    z_score = (spread - spread.mean()) / spread.std()
    data1['Signal'] = np.where(z_score > 1, -1, np.where(z_score < -1, 1, 0))
    data1['Returns'] = data1['Close'].pct_change()
    data1['Strategy_Returns'] = data1['Signal'].shift(1) * data1['Returns']
    return data1['Strategy_Returns'].cumsum().iloc[-1]

# Function to get buy/sell/hold recommendation
def get_recommendation(data):
    last_close = data['Close'].iloc[-1]
    sma_20 = data['SMA20'].iloc[-1]
    sma_120 = data['SMA120'].iloc[-1]
    rsi = data['RSI'].iloc[-1]
    
    if last_close > sma_20 and last_close > sma_120 and rsi < 70:
        return "Buy"
    elif last_close < sma_20 and last_close < sma_120 and rsi > 30:
        return "Sell"
    else:
        return "Hold"

# Function to get market sentiment
def get_market_sentiment(data):
    returns = data['Close'].pct_change().dropna()
    recent_returns = returns.tail(10)  # Look at last 10 days
    if recent_returns.mean() > 0:
        return "Bullish 🐂"
    else:
        return "Bearish 🐻"

# Function to create marquee ticker
def create_marquee_ticker():
    tickers = yf.Tickers(' '.join(popular_stocks))
    ticker_data = tickers.download(period='1d')['Close']
    
    ticker_text = ""
    for stock in popular_stocks:
        price = ticker_data[stock].iloc[-1]
        change = (price - ticker_data[stock].iloc[0]) / ticker_data[stock].iloc[0] * 100
        color = "green" if change >= 0 else "red"
        arrow = "▲" if change >= 0 else "▼"
        ticker_text += f"{stock}: ${price:.2f} <span style='color:{color};'>{arrow} ({change:.2f}%)</span> | "
    
    return ticker_text.strip('| ')

# Function to predict future returns
def predict_future_returns(data, investment_amount, days=[5, 10, 20, 30, 40, 50, 60]):
    try:
        df = data.reset_index()
        df['Date'] = (df['Date'] - df['Date'].min()).dt.days
        
        features = ['Date', 'SMA20', 'EMA20', 'RSI', 'VWAP']
        
        df_clean = df.dropna()
        X = df_clean[features]
        y = df_clean['Close']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # Calculate model accuracy
        y_pred = model.predict(X_test_scaled)
        model_accuracy = r2_score(y_test, y_pred)
        
        last_date = X['Date'].iloc[-1]
        future_dates = np.array([[last_date + i for i in days]])
        X_scaled = scaler.transform(X)
        future_features = np.hstack([future_dates.T, X_scaled[-1:, 1:].repeat(len(days), axis=0)])
        future_prices = model.predict(future_features)
        
        last_price = y.iloc[-1]
        future_returns = [(price / last_price - 1) for price in future_prices]
        expected_values = [investment_amount * (1 + return_) for return_ in future_returns]
        
        return dict(zip(days, expected_values)), model_accuracy
    
    except Exception as e:
        logging.error(f"Error in predict_future_returns: {str(e)}")
        st.error(f"An error occurred while predicting future returns: {str(e)}")
        return {}, 0

# Main Streamlit app
def main():
    st.title('Advanced Stock Analysis App')

    # Marquee ticker
    marquee_text = create_marquee_ticker()
    st.markdown(
        f'<div class="marquee"><span>{marquee_text}</span></div>',
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <style>
        .marquee {
            width: 100%;
            overflow: hidden;
            background: #f0f0f0;
            padding: 10px 0;
        }
        .marquee span {
            display: inline-block;
            width: max-content;
            padding-left: 100%;
            animation: marquee 60s linear infinite;
            font-family: Arial, sans-serif;
            font-size: 14px;
        }
        @keyframes marquee {
            0% {transform: translate(0, 0);}
            100% {transform: translate(-100%, 0);}
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Sidebar for user input
    st.sidebar.header('User Input')
    
    # Dropdown for stocks and indices
    analysis_type = st.sidebar.radio("Select Analysis Type", ('Popular Stocks', 'S&P 500 Sectors', 'Custom Stock'))
    
    if analysis_type == 'Popular Stocks':
        symbol = st.sidebar.selectbox('Select a Stock', popular_stocks)
    elif analysis_type == 'S&P 500 Sectors':
        sector = st.sidebar.selectbox('Select a Sector', list(sp500_sectors.keys()))
        symbol = sp500_sectors[sector]
    else:
        symbol = st.sidebar.text_input('Enter Stock Symbol', 'AAPL')
    
    start_date = st.sidebar.date_input('Start Date', pd.to_datetime('2020-01-01'))
    end_date = st.sidebar.date_input('End Date', pd.to_datetime('today'))
    
    indicators = st.sidebar.multiselect('Select Technical Indicators', 
                                        ['SMA20', 'SMA120', 'EMA20', 'RSI', 'VWAP'], 
                                        default=['SMA20', 'SMA120'])

    # Fetch data
    data = get_stock_data(symbol, start_date, end_date)
    
    if not data.empty:
        data = add_indicators(data)

        st.plotly_chart(plot_stock_data(data, indicators))

        # Display recommendation and sentiment
        col1, col2 = st.columns(2)
        with col1:
            recommendation = get_recommendation(data)
            st.subheader(f"Recommendation: {recommendation}")
        with col2:
            sentiment = get_market_sentiment(data)
            st.subheader(f"Market Sentiment: {sentiment}")

        # Display raw data
        if st.checkbox('Show Raw Data'):
            st.subheader('Raw Data')
            st.write(data)

        # Trading Strategies
        st.subheader('Trading Strategies')
        trends_momentum_return = trends_momentum_strategy(data)
        mean_reversion_return = mean_reversion_strategy(data)
        vwap_return = vwap_strategy(data)

        st.write(f"Trends and Momentum Strategy Return: {trends_momentum_return:.2%}")
        st.write(f"Mean Reversion Strategy Return: {mean_reversion_return:.2%}")
        st.write(f"VWAP Strategy Return: {vwap_return:.2%}")

        # Statistical Arbitrage (example with another stock)
        other_symbol = st.selectbox('Select another stock for Statistical Arbitrage', popular_stocks)
        other_data = get_stock_data(other_symbol, start_date, end_date)
        if not other_data.empty:
            stat_arb_return = statistical_arbitrage_strategy(data, other_data)
            st.write(f"Statistical Arbitrage Strategy Return: {stat_arb_return:.2%}")

        # Future Returns Prediction
        st.subheader('Future Returns Prediction')
        st.write("Note: This prediction is based on historical data and should not be the sole basis for investment decisions.")
        investment_amount = st.number_input('Enter Investment Amount ($)', min_value=1, value=1000)
        
        with st.spinner('Calculating future returns...'):
            future_returns, model_accuracy = predict_future_returns(data, investment_amount)
        
        if future_returns:
            st.write("Predicted future values of your investment:")
            for days, value in future_returns.items():
                st.write(f"{days} days: ${value:.2f}")
            st.write(f"Model Accuracy: {model_accuracy:.2%}")
        else:
            st.write("Unable to predict future returns at this time.")

        # Price Alerts (simulated)
        st.subheader('Price Alerts')
        alert_price = st.number_input('Set Alert Price', value=data['Close'].iloc[-1])
        if st.button('Set Alert'):
            st.write(f'Alert set for {symbol} at ${alert_price:.2f}')

        # News Feed (simulated)
        st.subheader('Recent News')
        st.write("This would typically integrate with a news API. For demonstration, we're showing placeholder news.")
        st.write("1. Market Update: S&P 500 Reaches New High")
        st.write(f"2. {symbol} Announces Quarterly Earnings")
        st.write("3. Federal Reserve Holds Interest Rates Steady")

    else:
        st.write('No data found for the given symbol and date range.')

if __name__ == '__main__':
    main()
