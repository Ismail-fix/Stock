import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import pandas as pd
import numpy as np
import pytz
import wikipediaapi
import tweepy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Configuration du fuseau horaire souhaité
TIMEZONE = 'Europe/Paris'

START = "2017-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.markdown("## 📈 Stock Prediction Dashboard")

company_tickers = {
    "Apple Inc.": "AAPL",
    "Alphabet Inc. (Google)": "GOOG",
    "Microsoft Corporation": "MSFT",
    "GameStop Corp.": "GME",
    "NVIDIA Corporation": "NVDA",
    "S&P 500 Index": "^GSPC",
    "VIX Index": "^VIX",
    "iShares Russell 2000 ETF": "IWM",
    "LVMH Moët Hennessy Louis Vuitton (MC.PA)": "MC.PA",
    "Dell Technologies Inc.": "DELL",
    "Atos SE":"ATO.PA",
    "Amazon.com, Inc.":"AMZN",
    "Bitcoin USD":"BTC-USD",
    "Cassava Sciences, Inc.":"SAVA",
    "NASDAQ Composite":"^IXIC",
    "Lumen Technologies Inc.":"LUMN",
    "Uber Technologies, Inc.":"UBER",
    "Super Micro Computer, Inc.":"SMCI",
    "Tesla, Inc.":"TSLA",
    "Intel Corporation":"INTC",
    "Taiwan Semiconductor Manufacturing Company Limited":"TSM",
    "MicroStrategy Incorporated":"MSTR",
    "Cambria Tail Risk ETF":"TAIL",
    "Palantir Technologies Inc.":"PLTR",
    "Take-Two Interactive Software, Inc.":"TTWO",
    "Meta Platforms, Inc.":"META",
    "Ethereum USD":"ETH-USD",
}

# Sidebar for settings
st.sidebar.title("🔧 Settings")
company_names = sorted(list(company_tickers.keys()))
selected_company = st.sidebar.selectbox("Select a company for prediction", company_names)
n_years = st.sidebar.slider("Years of prediction:", 1, 4)
selected_ticker = company_tickers[selected_company]
period = n_years * 252

# Fetch and display company description
def get_wikipedia_summary(company_name):
    wiki_wiki = wikipediaapi.Wikipedia(
        language='en',
        user_agent="StokPred/1.0 (Local Development; ismail.benotmane02@gmail.com)"
    )
    page = wiki_wiki.page(company_name)
    if page.exists():
        summary = page.summary.split('\n')[0]
        return summary
    else:
        return "No Wikipedia summary available for this company."

company_description = get_wikipedia_summary(selected_company.split(" (")[0])
st.sidebar.write(f"**{selected_company}**: {company_description}")

# Load and process data
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    
    # Ensure 'Date' column is in datetime format
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

    # Now you can safely use .dt accessor
    if data['Date'].dt.tz is None:
        data['Date'] = data['Date'].dt.tz_localize('UTC', ambiguous='NaT')
    else:
        data['Date'] = data['Date'].dt.tz_convert(TIMEZONE)
    
    data['Daily Return'] = data['Close'].pct_change()
    data['Volatility'] = data['Daily Return'].rolling(window=30).std() * np.sqrt(252)
    return data


@st.cache_data(ttl=60)
def load_live_data(ticker):
    live_data = yf.download(ticker, period="1d", interval="1m")
    live_data.reset_index(inplace=True)
    if live_data['Datetime'].dt.tz is None:
        live_data['Datetime'] = live_data['Datetime'].dt.tz_localize('UTC', ambiguous='NaT')
    else:
        live_data['Datetime'] = live_data['Datetime'].dt.tz_convert(TIMEZONE)
    return live_data

data = load_data(selected_ticker)
live_data = load_live_data(selected_ticker)

# Tabs for organizing content
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Historical Data", "Live Data", "Prophet Forecast", "Linear Regression", 
    "Random Forest", "LSTM Prediction", "Sentiment Analysis"])

# Historical Data
with tab1:
    st.subheader('Historical Data')
    st.write(data.tail())
    
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='stock_close', line=dict(color='blue')))
        fig.update_layout(title="Historical Data", xaxis_title="Date", yaxis_title="Close Price", xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_raw_data()

# Live Data
with tab2:
    st.subheader('Live Data')
    st.write(live_data.tail())
    
    def plot_live_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=live_data['Datetime'], y=live_data['Close'], mode='lines', name='live_stock_close', line=dict(color='green')))
        fig.update_layout(title="Live Data", xaxis_title="Datetime", yaxis_title="Close Price", xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
    
    plot_live_data()

# Prophet Forecast
with tab3:
    st.subheader('Prophet Forecast')
    
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
    df_train['ds'] = df_train['ds'].dt.tz_localize(None)

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    st.write(forecast.tail())
    
    st.write('Prophet forecast')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    # Recommendation based on actual vs predicted price
    latest_actual_price = live_data['Close'].iloc[-1]
    latest_forecast = forecast.iloc[-1]

    if latest_actual_price < latest_forecast['yhat_lower']:
        recommendation = "BUY"
    elif latest_actual_price > latest_forecast['yhat_upper']:
        recommendation = "SELL"
    else:
        recommendation = "HOLD"

    st.subheader('Recommendation')
    st.write(f"The current recommendation is: **{recommendation}**")

# Linear Regression
with tab4:
    st.subheader('Linear Regression Prediction')
    
    def linear_regression_prediction(data, period):
        X = np.array(range(len(data))[-period:]).reshape(-1, 1)
        y = data['Close'].iloc[-period:]
        model = LinearRegression().fit(X, y)
        future_X = np.array(range(len(data), len(data) + period)).reshape(-1, 1)
        prediction = model.predict(future_X)
        return prediction

    lr_prediction = linear_regression_prediction(data, period)
    
    future_dates = pd.date_range(start=data['Date'].iloc[-1], periods=period).tolist()

    def plot_model_predictions(data, period, future_dates, model_name, prediction):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Actual Price'))
        fig.add_trace(go.Scatter(x=future_dates, y=prediction, name=f'{model_name} Prediction'))
        fig.layout.update(title_text=f"{model_name} Prediction", xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_model_predictions(data, period, future_dates, "Linear Regression", lr_prediction)

# Random Forest Prediction
with tab5:
    st.subheader('Random Forest Prediction')
    
    def random_forest_prediction(data, period):
        X = data[['Open', 'High', 'Low', 'Close', 'Volume', 'Volatility']].iloc[:-period]
        y = data['Close'].iloc[period:]
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        X_future = data[['Open', 'High', 'Low', 'Close', 'Volume', 'Volatility']].iloc[-period:]
        prediction = model.predict(X_future)
        return prediction

    rf_prediction = random_forest_prediction(data, period)

    plot_model_predictions(data, period, future_dates, "Random Forest", rf_prediction)

# LSTM Prediction
with tab6:
    st.subheader('LSTM Prediction')
    
    def lstm_prediction(data, period):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data[['Close']].values)

        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(scaled_data.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the LSTM model
        model.fit(scaled_data[:-period], scaled_data[1:-period+1], epochs=50, batch_size=32)

        # Make predictions for the specified period
        inputs = scaled_data[-period:]
        inputs = inputs.reshape(-1, 1)
        prediction = []
        for _ in range(period):
            inputs = np.reshape(inputs, (1, inputs.shape[0], 1))
            output = model.predict(inputs)
            prediction.append(output[0, 0])
            inputs = np.append(inputs[0], output)
        prediction = scaler.inverse_transform(np.array(prediction).reshape(-1, 1))

        return prediction.flatten()

    lstm_prediction_values = lstm_prediction(data, period)

    plot_model_predictions(data, period, future_dates, "LSTM", lstm_prediction_values)

# Market Sentiment Analysis
with tab7:
    st.subheader("Market Sentiment Tracker")

    # Twitter API credentials provided by you
consumer_key = "HRv23IsKFOk5ZYvsH9cNwXWd5"
consumer_secret = "xc58tXNcRSiCxxpGnRDokbqxiYE8NzhZWpJfmRml9FDe7JYaNb"
access_token = "563835535-bdsTa88xE3Dl7mwlz1j9W7CaZPbc3RYAxyXbHY4D"
access_token_secret = "zQDjMoN6LiN672H51t3S7eBsLLn4M44RtNMPBvS9mI4tN"

# Set up Tweepy client with OAuth 1.1a (Note: Access to v1.1 might be limited)
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

# Fetch tweets function using Twitter API v2 (via Bearer Token)
def fetch_tweets_v2(query, max_results=10):
    try:
        client = tweepy.Client(bearer_token="AAAAAAAAAAAAAAAAAAAAABUMvQEAAAAAqXXPvc2%2BnZ%2BPopylHa9fkh9QTUY%3D7yyRvVwi3aqylecwU8dHEmASoN29vyRKYLSAOXVw0LWxqLqR6X")
        tweets = client.search_recent_tweets(query=query, max_results=max_results, tweet_fields=['text', 'created_at'])
        tweet_texts = [tweet.text for tweet in tweets.data] if tweets.data else []
        return tweet_texts
    except tweepy.TweepyException as e:
        st.error(f"An error occurred: {e}")
        return []

# Analyze sentiment function
def analyze_sentiment(tweets):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = {"positive": 0, "neutral": 0, "negative": 0}

    for tweet in tweets:
        sentiment = analyzer.polarity_scores(tweet)
        if sentiment["compound"] >= 0.05:
            sentiment_scores["positive"] += 1
        elif sentiment["compound"] <= -0.05:
            sentiment_scores["negative"] += 1
        else:
            sentiment_scores["neutral"] += 1

    total = sum(sentiment_scores.values())
    for key in sentiment_scores:
        sentiment_scores[key] = sentiment_scores[key] / total

    return sentiment_scores

# Streamlit interface for Market Sentiment Tracker
st.subheader("Market Sentiment Tracker")

# Using the selected company name for Twitter query
company_query = "Apple"  # Example company; you can replace this with a dynamic query from your app
tweets = fetch_tweets_v2(company_query)
if tweets:
    sentiment_scores = analyze_sentiment(tweets)

    st.write(f"Sentiment Analysis for {company_query}:")
    st.write(f"Positive: {sentiment_scores['positive']:.2%}")
    st.write(f"Neutral: {sentiment_scores['neutral']:.2%}")
    st.write(f"Negative: {sentiment_scores['negative']:.2%}")

    st.bar_chart(pd.DataFrame.from_dict(sentiment_scores, orient='index', columns=['Sentiment']))
else:
    st.write("No tweets found or there was an error fetching tweets.")
