from flask import Flask, jsonify
import pyodbc
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
import requests
import json
import os
import logging

app = Flask(__name__)

# load_dotenv()

# Setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_KEY = 'VdtFjg7ouEX7v8DD2vSayolTYWFHR5GP'

def init_db():
    conn = None
    try:
        server = os.environ['AZURE_DB_SERVER']
        database = os.environ['AZURE_DB_NAME']
        username = os.environ['AZURE_DB_USERNAME']
        password = os.environ['AZURE_DB_PASSWORD']

        driver = '{ODBC Driver 17 for SQL Server}'

        conn = pyodbc.connect(f'DRIVER={driver};SERVER=tcp:{server};PORT=1433;DATABASE={database};UID={username};PWD={password}')

        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY,
                date TEXT NOT NULL,
                ticker TEXT NOT NULL,
                drn_signal TEXT,
                power_buy_signal TEXT,
                vwap_signal TEXT,
                vwma_signal TEXT,
                rsi_signal TEXT,
                fibonacci_signal TEXT,
                date_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    except pyodbc.Error as e:
        logger.error(f"Database connection error: {e}")
    finally:
        if conn:
            conn.close()

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Welcome to the Stock Prediction API'}), 200

@app.route('/<string:ticker>', methods=['GET'])
def predict_stock(ticker):
    try:
        data = fetch_stock_data(ticker)
        if data is not None:
            df = pd.DataFrame(data)
            prediction = calculate_all_signals(df)
            save_prediction_to_db(ticker, prediction)
            return jsonify({'ticker': ticker, 'prediction': prediction})
        else:
            return jsonify({'error': 'Failed to fetch stock data'}), 500
    except Exception as e:
        logger.error(f"An error occurred while predicting stock for {ticker}: {e}", exc_info=True)
        return jsonify({'error': 'An unexpected error occurred'}), 500


def fetch_stock_data(ticker):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=120)

    formatted_start_date = start_date.strftime('%Y-%m-%d')
    formatted_end_date = end_date.strftime('%Y-%m-%d')

    url = f"https://fmpcloud.io/api/v3/historical-price-full/{ticker}?from={formatted_start_date}&to={formatted_end_date}&apikey=0a737c900d6e11f970871530deb732b6"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = json.loads(response.text)
        return data.get('historical', [])
    except requests.RequestException as e:
        logger.error(f"Error during requests to {url}: {e}")
        return None

def save_prediction_to_db(ticker, prediction):
    try:
        server = os.environ['AZURE_DB_SERVER']
        database = os.environ['AZURE_DB_NAME']
        username = os.environ['AZURE_DB_USERNAME']
        password = os.environ['AZURE_DB_PASSWORD']
        driver = '{ODBC Driver 17 for SQL Server}'

        conn = pyodbc.connect(f'DRIVER={driver};SERVER=tcp:{server};PORT=1433;DATABASE={database};UID={username};PWD={password}')
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions 
            (date, ticker, drn_signal, power_buy_signal, vwap_signal, vwma_signal, rsi_signal, fibonacci_signal) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            prediction['date'],
            ticker,
            prediction['drn_signal'],
            prediction['power_buy_signal'],
            prediction['vwap_signal'],
            prediction['vwma_signal'],
            prediction['rsi_signal'],
            prediction['fibonacci_signal']
        ))
        conn.commit()
    except pyodbc.Error as e:
        logger.error(f"Error saving prediction to the database: {e}")
    finally:
        if conn:
            conn.close()

# Include your calculate_all_signals function here
def calculate_all_signals(df):
    # DRN Shortest RSI Signals
    ema1_length, ema2_length, ema3_length = 3, 5, 8
    df['drn'] = (df['high'] - df['low'] + df['close']).ewm(span=ema1_length, adjust=False).mean() - \
                (df['high'] - df['low'] + df['close']).ewm(span=ema2_length, adjust=False).mean() + \
                (df['high'] - df['low'] + df['close']).ewm(span=ema3_length, adjust=False).mean()
    df['is_prev_smaller'] = df['drn'] > df['drn'].shift(-1)
    df['drn_signal'] = df.apply(lambda row: 'Buy' if row['is_prev_smaller'] else 'Sell', axis=1)

    # Custom Power Buy Signals
    df['avg_volume'] = df['volume'].rolling(window=5).mean()
    buy_conditions = (
        (df['volume'] > df['avg_volume']) & (df['close'] > df['open']) |  # Volume Breakout
        ((df['volume'] > df['avg_volume']) & (df['close'] > df['open'])) |  # High Volume Green
        (df['close'] < df['close'].shift(1)) & (df['volume'] < df['volume'].shift(1))  # Buy Signal Pullback
    )
    df['power_buy_signal'] = buy_conditions.map({True: 'Buy', False: 'Sell'})

    # VWAP Signal
    df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    df['vwap_signal'] = np.where(
        (df['close'] < df['vwap']) & (df['high'] > df['low'].shift(1)) & (df['close'] > df['low'].shift(1)), 'Buy',
        np.where(
            (df['close'] > df['vwap']) & (df['low'] < df['high'].shift(1)) & (df['close'] < df['high'].shift(1)), 'Sell', '--'))

    # VWMA Signal
    length = 4
    vwma = (df['close'] * df['volume']).rolling(window=length).mean() / df['volume'].rolling(window=length).mean()
    roc_vwma = vwma - vwma.shift(1)
    roc_sma_vwma = vwma.rolling(window=4).mean() - vwma.rolling(window=4).mean().shift(1)
    roc_of_roc_sma_vwma = roc_sma_vwma - roc_sma_vwma.shift(1)
    df['vwma_signal'] = ['Buy' if roc_vwma.iloc[idx] > roc_of_roc_sma_vwma.iloc[idx] else '--' for idx in df.index]

    # RSI Signal
    rsi_length, up_level, down_level = 2, 70, 30
    delta = df['close'].diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    avg_gain = up.rolling(window=rsi_length, min_periods=1).mean()
    avg_loss = down.abs().rolling(window=rsi_length, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    df['rsi'] = rsi
    df['rsi_signal'] = np.where(df['rsi'] > up_level, '--', np.where(df['rsi'] < down_level, 'Buy', '--'))

    # Fibonacci Signal
    length = 55
    df['level_1'] = df['high'].rolling(window=length).max()
    df['level_0'] = df['low'].rolling(window=length).min()
    df['range'] = df['level_1'] - df['level_0']
    df['level_786'] = df['level_1'] - df['range'] * 0.786
    df['fibonacci_signal'] = df.apply(lambda row: 'Buy' if row['low'] <= row['level_786'] and row['close'] > row['open'] else '--', axis=1)

    return df[['date', 'drn_signal', 'power_buy_signal', 'vwap_signal', 'vwma_signal', 'rsi_signal', 'fibonacci_signal']].iloc[0].to_dict()

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
