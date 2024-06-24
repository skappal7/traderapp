import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas_ta as ta
import numpy as np
from datetime import datetime, timedelta

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
def add_indicators(data, indicators):
    if 'SMA' in indicators:
        data['SMA'] = ta.sma(data['Close'], length=20)
    if 'EMA' in indicators:
        data['EMA'] = ta.ema(data['Close'], length=20)
    if 'RSI' in indicators:
        data['RSI'] = ta.rsi(data['Close'], length=14)
    return data

# Function for simple backtesting
def simple_backtest(data, strategy='SMA Crossover'):
    if strategy == 'SMA Crossover':
        data['SMA_short'] = ta.sma(data['Close'], length=10)
        data['SMA_long'] = ta.sma(data['Close'], length=30)
        data['Position'] = np.where(data['SMA_short'] > data['SMA_long'], 1, 0)
        data['Returns'] = data['Close'].pct_change()
        data['Strategy_Returns'] = data['Position'].shift(1) * data['Returns']
        cumulative_returns = (1 + data['Strategy_Returns']).cumprod()
        return cumulative_returns.iloc[-1] - 1

# Function to get buy/sell/hold recommendation
def get_recommendation(data):
    last_close = data['Close'].iloc[-1]
    sma_20 = ta.sma(data['Close'], length=20).iloc[-1]
    sma_50 = ta.sma(data['Close'], length=50).iloc[-1]
    rsi = ta.rsi(data['Close'], length=14).iloc[-1]
    
    if last_close > sma_20 and last_close > sma_50 and rsi < 70:
        return "Buy"
    elif last_close < sma_20 and last_close < sma_50 and rsi > 30:
        return "Sell"
    else:
        return "Hold"

# Function to get market sentiment
def get_market_sentiment(data):
    returns = data['Close'].pct_change().dropna()
    recent_returns = returns.tail(10)  # Look at last 10 days
    if recent_returns.mean() > 0:
        return "Bullish"
    else:
        return "Bearish"

# Function to create marquee ticker
def create_marquee_ticker():
    tickers = yf.Tickers(' '.join(popular_stocks))
    ticker_data = tickers.download(period='1d')['Close']
    
    ticker_text = ""
    for stock in popular_stocks:
        price = ticker_data[stock].iloc[-1]
        change = (price - ticker_data[stock].iloc[0]) / ticker_data[stock].iloc[0] * 100
        ticker_text += f"{stock}: ${price:.2f} ({change:.2f}%) | "
    
    return ticker_text.strip('| ')

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
                                        ['SMA', 'EMA', 'RSI'], 
                                        default=['SMA'])

    # Fetch data
    data = get_stock_data(symbol, start_date, end_date)
    
    if not data.empty:
        # Add indicators
        data = add_indicators(data, indicators)

        # Plot stock data
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

        # Backtesting
        st.subheader('Simple Backtesting')
        backtest_result = simple_backtest(data)
        st.write(f'Strategy Return: {backtest_result:.2%}')

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
