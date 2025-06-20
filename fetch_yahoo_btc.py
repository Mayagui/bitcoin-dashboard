import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Période : 5 ans glissants
end = datetime.now()
start = end - timedelta(days=5*365)

ticker = 'BTC-USD'
df = yf.download(ticker, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), interval='1d')

if not df.empty:
    df = df.reset_index()
    df = df.rename(columns={
        'Date': 'open_time',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })
    df['close_time'] = df['open_time'] + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    df = df[['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time']]
    df.to_csv('btc_ohlcv.csv', index=False)
    print(f"Historique BTC/USDT Yahoo Finance téléchargé ({start.date()} -> {end.date()}) dans btc_ohlcv.csv")
else:
    print("Erreur : impossible de télécharger l'historique BTC/USDT depuis Yahoo Finance.")
